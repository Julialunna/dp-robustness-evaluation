"""Training utilities for MNIST FL experiments.

This version implements the split proposed in the conversation:

1. Reserve PRETRAIN_RATIO of the MNIST training set as a public/auxiliary subset.
2. Train a small MLP on this public subset and save only its embedding extractor.
3. Use the remaining examples as the private/federated subset partitioned across clients.
4. Each client extracts embeddings with the frozen pretrained MLP, trains a local DP-CVAE,
   generates synthetic embeddings, and trains the federated downstream classifier.

Privacy note: the DP guarantee applies to the private/federated subset. The public/auxiliary
pretraining subset is intentionally excluded from client partitions.
"""

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


class EmbeddingMLP(nn.Module):
    """MLP used to map MNIST images into a shared embedding space.

    It is pretrained on the public/auxiliary subset and then frozen on every
    client/server. The final Sigmoid keeps embeddings in [0, 1], which matches
    the Sigmoid output of the CVAE decoder.
    """

    def __init__(self, input_size: int = 28 * 28, hidden_size: int = 256, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


class EmbeddingPretrainClassifier(nn.Module):
    """Temporary model used only to pretrain the EmbeddingMLP."""

    def __init__(self, hidden_size: int, embedding_dim: int, num_classes: int):
        super().__init__()
        self.extractor = EmbeddingMLP(hidden_size=hidden_size, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extractor(x)
        return self.classifier(emb)


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


# -----------------------------------------------------------------------------
# Public MLP extractor pretraining
# -----------------------------------------------------------------------------
def pretrain_embedding_extractor(
    *,
    force: bool | None = None,
    device: torch.device | None = None,
) -> Path:
    """Train the MLP extractor on the public/auxiliary MNIST subset and save it.

    This is intentionally done before Flower starts. The saved extractor is then
    frozen and shared by server/clients.
    """
    path = Path(parameters_federated.EMBEDDING_EXTRACTOR_PATH)
    if force is None:
        force = bool(parameters_federated.FORCE_RETRAIN_EXTRACTOR)

    if path.exists() and not force:
        print(f"[Pretrain] Extrator já existe em {path}. Pulando pré-treino.")
        return path

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    public_dataset, _ = get_public_private_mnist_splits()
    if len(public_dataset) == 0:
        raise RuntimeError("PRETRAIN_RATIO gerou 0 exemplos públicos; aumente PRETRAIN_RATIO.")

    train_data = _apply_transforms_to_dataset(public_dataset)
    train_loader = DataLoader(
        train_data,
        batch_size=parameters_federated.PRETRAIN_BATCH_SIZE,
        shuffle=True,
    )

    torch.manual_seed(parameters_federated.EMBEDDING_EXTRACTOR_SEED)
    model = EmbeddingPretrainClassifier(
        hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
        embedding_dim=parameters_federated.EMBEDDING_DIM,
        num_classes=parameters_federated.NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters_federated.PRETRAIN_LR,
        weight_decay=parameters_federated.PRETRAIN_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    print(
        f"[Pretrain] Treinando MLP extratora com {len(public_dataset)} exemplos públicos "
        f"por {parameters_federated.PRETRAIN_EPOCHS} épocas..."
    )
    for epoch in range(parameters_federated.PRETRAIN_EPOCHS):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        print(f"[Pretrain] Época {epoch + 1}/{parameters_federated.PRETRAIN_EPOCHS} | loss={avg_loss:.4f} | acc={acc:.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.extractor.state_dict(), path)
    meta_path = path.with_suffix(path.suffix + ".meta.pt")
    torch.save(
        {
            "pretrain_ratio": parameters_federated.PRETRAIN_RATIO,
            "data_split_seed": parameters_federated.DATA_SPLIT_SEED,
            "num_public_examples": len(public_dataset),
            "embedding_dim": parameters_federated.EMBEDDING_DIM,
            "hidden_size": parameters_federated.EMBEDDING_HIDDEN_SIZE,
            "epochs": parameters_federated.PRETRAIN_EPOCHS,
        },
        meta_path,
    )
    print(f"[Pretrain] Extrator salvo em {path}")
    return path


def ensure_embedding_extractor_ready() -> None:
    if parameters_federated.PRETRAIN_EXTRACTOR:
        pretrain_embedding_extractor()
    else:
        path = getattr(parameters_federated, "EMBEDDING_EXTRACTOR_PATH", None)
        if path and not Path(path).exists():
            print(
                f"[Aviso] PRETRAIN_EXTRACTOR=False e {path} não existe. "
                "O código usará uma MLP aleatória congelada."
            )


def build_embedding_extractor(device: torch.device) -> EmbeddingMLP:
    """Create the same frozen MLP embedding extractor on every client/server."""
    torch.manual_seed(parameters_federated.EMBEDDING_EXTRACTOR_SEED)
    model = EmbeddingMLP(
        hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
        embedding_dim=parameters_federated.EMBEDDING_DIM,
    )

    path = getattr(parameters_federated, "EMBEDDING_EXTRACTOR_PATH", None)
    if path and Path(path).exists():
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    elif path:
        print(f"[Aviso] Extrator pré-treinado não encontrado em {path}. Usando MLP aleatória congelada.")

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# -----------------------------------------------------------------------------
# Embedding extraction and downstream classifier training/evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings_from_loader(
    embedding_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a MNIST dataloader into tensors of embeddings and labels."""
    embedding_model.to(device)
    embedding_model.eval()
    all_embeddings, all_labels = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].long()
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
