"""
Módulo de Carregamento e Particionamento de Dados.

Gerencia o download, transformações de imagem e divisão federada (Dirichlet / IID)
utilizando Flower Datasets (flwr-datasets).
"""

import os
from typing import Tuple

import torch
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

import config

# Evita timeouts e verificações de rede no Hugging Face Hub se a conexão cair
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

_CACHED_FDS = {}


def get_dataset_name(dataset: str = None) -> str:
    """Retorna o identificador Hugging Face do dataset."""
    ds = dataset or config.DATASET
    if ds == "mnist":
        return "ylecun/mnist"
    return f"danjacobellis/{ds}_224"


def _flatten_labels(batch):
    """Garante que os rótulos sejam inteiros escalares 1D."""
    batch["label"] = [
        int(label[0]) if isinstance(label, list) else int(label)
        for label in batch["label"]
    ]
    return batch


def _apply_transforms(batch):
    """Converte imagens PIL em tensores PyTorch."""
    transform = Compose([ToTensor()])
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


def get_client_loaders(
    partition_id: int,
    num_partitions: int = config.NUM_PARTITIONS,
) -> Tuple[DataLoader, DataLoader]:
    """
    Carrega as partições de treino e teste para um cliente federado específico.
    
    100% dos dados de treino são distribuídos entre os clientes via Dirichlet ou IID.
    """
    global _CACHED_FDS

    cache_key = (config.DATASET, config.PARTITIONER, config.DIRICHLET_ALPHA, num_partitions)

    if cache_key not in _CACHED_FDS:
        if config.PARTITIONER == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=config.DIRICHLET_ALPHA,
                min_partition_size=10,
                self_balancing=True,
                shuffle=True,
            )
            validation_partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=config.DIRICHLET_ALPHA,
                min_partition_size=10,
                self_balancing=True,
                shuffle=True,
            )
        elif config.PARTITIONER == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
            validation_partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            raise ValueError(f"Particionador '{config.PARTITIONER}' não suportado.")

        partitioners = {"train": partitioner}
        if config.DATASET != "mnist":
            partitioners["validation"] = validation_partitioner

        _CACHED_FDS[cache_key] = FederatedDataset(
            dataset=get_dataset_name(),
            partitioners=partitioners,
            preprocessor=lambda ds: ds.map(_flatten_labels, batched=True),
        )

    fds = _CACHED_FDS[cache_key]

    # Carrega partição de treino local do cliente
    train_partition = fds.load_partition(
        partition_id, split="train"
    ).with_transform(_apply_transforms)

    # Carrega partição de teste/validação local do cliente
    if config.DATASET == "mnist":
        train_test_split = train_partition.train_test_split(test_size=0.2, seed=42)
        train_partition = train_test_split["train"]
        test_partition = train_test_split["test"]
    else:
        test_partition = fds.load_partition(
            partition_id, split="validation"
        ).with_transform(_apply_transforms)

    train_loader = DataLoader(
        train_partition,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_partition,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


def get_server_test_loader(dataset_name: str = None) -> DataLoader:
    """Carrega o conjunto completo de teste para avaliação centralizada no servidor."""
    name = dataset_name or get_dataset_name()
    try:
        dataset = load_dataset(name, local_files_only=True)
    except Exception:
        dataset = load_dataset(name)

    test_data = dataset["test"].map(_flatten_labels, batched=True)
    test_data = test_data.with_transform(_apply_transforms)

    return DataLoader(
        test_data,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )


def reset_federated_dataset() -> None:
    """Reseta o cache global do FederatedDataset entre repetições do experimento."""
    global _CACHED_FDS
    _CACHED_FDS.clear()

