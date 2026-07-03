
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor

import parameters_federated

_mnist_split_cache: Dict[str, Dataset] = {}
_partition_indices_cache: Dict[Tuple[int, str, float, int], List[np.ndarray]] = {}


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    """Original image classifier: 28x28 image -> class logits."""

    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        y = self.relu1(self.fc1(x))
        y = self.relu2(self.fc2(y))
        return self.fc3(y)



class EmbeddingClassifier(nn.Module):
    """Federated downstream classifier: synthetic/real embedding -> class logits."""

    def __init__(self, input_size: int = 128, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# -----------------------------------------------------------------------------
# Weights helpers
# -----------------------------------------------------------------------------
def disable_inplace_relu(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def get_weights(net: nn.Module):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# -----------------------------------------------------------------------------
# Dataset split and partitioning
# -----------------------------------------------------------------------------
def _get_transforms():
    return Compose([ToTensor(), Normalize((parameters_federated.MEAN,), (parameters_federated.STD,))])


def _apply_transforms_to_dataset(dataset: Dataset) -> Dataset:
    pytorch_transforms = _get_transforms()

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    return dataset.with_transform(apply_transforms)


def get_public_private_mnist_splits() -> Tuple[Dataset, Dataset]:
    """Return non-overlapping public/pretrain and private/federated MNIST splits."""
    global _mnist_split_cache
    if "public" in _mnist_split_cache and "private" in _mnist_split_cache:
        return _mnist_split_cache["public"], _mnist_split_cache["private"]

    dataset = load_dataset("ylecun/mnist")
    train_dataset = dataset["train"].shuffle(seed=parameters_federated.DATA_SPLIT_SEED)

    pretrain_ratio = float(parameters_federated.PRETRAIN_RATIO)
    if not 0.0 <= pretrain_ratio < 1.0:
        raise ValueError("PRETRAIN_RATIO must be in [0, 1).")

    n_total = len(train_dataset)
    n_public = int(round(n_total * pretrain_ratio))

    public_dataset = train_dataset.select(range(0, n_public))
    private_dataset = train_dataset.select(range(n_public, n_total))

    _mnist_split_cache["public"] = public_dataset
    _mnist_split_cache["private"] = private_dataset

    print(
        f"[Dados] MNIST train dividido sem sobreposição: "
        f"{len(public_dataset)} públicos/pretrain ({pretrain_ratio:.0%}) + "
        f"{len(private_dataset)} privados/federados."
    )
    return public_dataset, private_dataset


def _build_iid_indices(num_examples: int, num_partitions: int) -> List[np.ndarray]:
    rng = np.random.default_rng(parameters_federated.DATA_SPLIT_SEED)
    indices = rng.permutation(num_examples)
    return [arr.astype(np.int64) for arr in np.array_split(indices, num_partitions)]


def _build_dirichlet_indices(labels: np.ndarray, num_partitions: int) -> List[np.ndarray]:
    """Partition private examples by class according to a Dirichlet distribution."""
    alpha = float(parameters_federated.DIRICHLET_ALPHA)
    rng = np.random.default_rng(parameters_federated.DATA_SPLIT_SEED)
    client_indices: List[List[int]] = [[] for _ in range(num_partitions)]

    for class_id in range(parameters_federated.NUM_CLASSES):
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_partitions, alpha))
        # Convert probabilities into split points over class_indices.
        split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
        class_splits = np.split(class_indices, split_points)
        for cid, split in enumerate(class_splits):
            client_indices[cid].extend(split.tolist())

    # Shuffle each client locally for deterministic but mixed ordering.
    result = []
    for cid, idxs in enumerate(client_indices):
        arr = np.array(idxs, dtype=np.int64)
        rng.shuffle(arr)
        result.append(arr)
        if len(arr) == 0:
            print(f"[Aviso] Cliente {cid} recebeu 0 exemplos. Considere aumentar DIRICHLET_ALPHA.")
    return result


def get_private_partition_indices(num_partitions: int) -> List[np.ndarray]:
    key = (
        num_partitions,
        parameters_federated.PARTITIONER,
        float(parameters_federated.DIRICHLET_ALPHA),
        int(parameters_federated.DATA_SPLIT_SEED),
    )
    if key in _partition_indices_cache:
        return _partition_indices_cache[key]

    _, private_dataset = get_public_private_mnist_splits()
    if parameters_federated.PARTITIONER == "iid":
        partitions = _build_iid_indices(len(private_dataset), num_partitions)
    elif parameters_federated.PARTITIONER == "dirichlet":
        labels = np.array(private_dataset["label"], dtype=np.int64)
        partitions = _build_dirichlet_indices(labels, num_partitions)
    else:
        raise ValueError("PARTITIONER must be either 'iid' or 'dirichlet'.")

    _partition_indices_cache[key] = partitions
    sizes = [len(p) for p in partitions]
    print(f"[Dados] Tamanhos dos clientes privados/federados: {sizes}")
    return partitions


def load_data(partition_id: int, num_partitions: int):
    """Load one client's private/federated split.

    The 10% public/pretrain subset is excluded here. Only the remaining private
    examples are partitioned among clients.
    """
    _, private_dataset = get_public_private_mnist_splits()
    partitions = get_private_partition_indices(num_partitions)
    client_indices = partitions[int(partition_id)]

    if len(client_indices) == 0:
        raise RuntimeError(
            f"Cliente {partition_id} recebeu 0 exemplos. "
            "Aumente DIRICHLET_ALPHA ou reduza NUM_PARTITIONS."
        )

    client_dataset = private_dataset.select(client_indices.tolist())
    split = client_dataset.train_test_split(test_size=0.2, seed=parameters_federated.DATA_SPLIT_SEED)
    train_data = _apply_transforms_to_dataset(split["train"])
    test_data = _apply_transforms_to_dataset(split["test"])

    train_loader = DataLoader(
        train_data,
        batch_size=parameters_federated.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=parameters_federated.BATCH_SIZE,
    )
    return train_loader, test_loader


def get_test_loader(dataset_str: str = "ylecun/mnist") -> DataLoader:
    dataset = load_dataset(dataset_str)
    test_data = _apply_transforms_to_dataset(dataset["test"])
    return DataLoader(test_data, batch_size=parameters_federated.BATCH_SIZE)




def add_gaussian_noise_to_normalized_images(
    images: torch.Tensor,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    
    if noise_std <= 0:
        return images

    mean = parameters_federated.MEAN
    std = parameters_federated.STD

    images_pixel = images * std + mean

    noise = torch.randn(
        images_pixel.shape,
        device=images_pixel.device,
        dtype=images_pixel.dtype,
        generator=generator,
    ) * noise_std

    noisy_pixel = images_pixel + noise
    noisy_pixel = torch.clamp(noisy_pixel, 0.0, 1.0)

    noisy_normalized = (noisy_pixel - mean) / std
    return noisy_normalized

@torch.no_grad()
def extract_embeddings_from_loader(
    embedding_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_noise_std: float = 0.0,
    image_noise_seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a MNIST dataloader into tensors of embeddings and labels."""
    embedding_model.to(device)
    embedding_model.eval()
    all_embeddings, all_labels = [], []
    generator = None
    if image_noise_std > 0 and image_noise_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(image_noise_seed)

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long()
        if image_noise_std > 0:
            images = add_gaussian_noise_to_normalized_images(
                images,
                image_noise_std,
                generator=generator,
            )
        embeddings = embedding_model(images).cpu()
        all_embeddings.append(embeddings)
        all_labels.append(labels.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def make_embedding_loader(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(embeddings.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_embedding_classifier(
    net: nn.Module,
    embedding_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
) -> None:
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()

    for _ in range(epochs):
        for embeddings, labels in embedding_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(embeddings), labels)
            loss.backward()
            optimizer.step()


def test_embedding_classifier_on_loader(
    net: nn.Module,
    embedding_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for embeddings, labels in embedding_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = net(embeddings)
            loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss / max(total, 1), correct / max(total, 1)


def test_embedding_classifier_from_images(
    net: nn.Module,
    embedding_model: nn.Module,
    image_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    embeddings, labels = extract_embeddings_from_loader(embedding_model, image_loader, device)
    loader = make_embedding_loader(
        embeddings,
        labels,
        batch_size=parameters_federated.BATCH_SIZE,
        shuffle=False,
    )
    return test_embedding_classifier_on_loader(net, loader, device)


# -----------------------------------------------------------------------------
# Original image-classifier training/testing kept for compatibility.
# -----------------------------------------------------------------------------
def train(net, train_loader, privacy_engine, optimizer, target_delta, device, epochs: int = 1):
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()

    for _ in range(epochs):
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            criterion(net(images), labels).backward()
            optimizer.step()

    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        return epsilon
    return None


def test(net, test_loader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / max(total, 1)
    return loss / max(total, 1), accuracy
