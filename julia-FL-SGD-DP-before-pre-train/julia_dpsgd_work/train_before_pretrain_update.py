"""Training utilities for MNIST FL experiments.

This file now supports two kinds of models:
1. The original image MLP classifier.
2. An embedding pipeline: image -> frozen MLP embedding extractor -> embedding classifier.

In the DP-CVAE experiment, Flower/FedAvg is applied to the embedding classifier,
not to the CVAE. Each client trains its own local DP-CVAE and uses the generated
synthetic embeddings to train the federated classifier.
"""

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

import parameters_federated

fds = None


class MLP(nn.Module):
    """Original image classifier: 28x28 image -> class logits."""

    def __init__(self, input_size=28 * 28, hidden_size=256, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.relu1(self.fc1(x))
        y = self.relu2(self.fc2(y))
        return self.fc3(y)


class EmbeddingMLP(nn.Module):
    """Frozen MLP used to map MNIST images into a shared embedding space.

    By default, this extractor is deterministic and shared across clients through
    a fixed random seed. For stronger experiments, replace it by a public or
    DP-trained pretrained MLP and load its state_dict in build_embedding_extractor.
    """

    def __init__(self, input_size=28 * 28, hidden_size=256, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class EmbeddingClassifier(nn.Module):
    """Federated downstream classifier: synthetic/real embedding -> class logits."""

    def __init__(self, input_size=128, hidden_size=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def build_embedding_extractor(device: torch.device) -> EmbeddingMLP:
    """Create the same MLP embedding extractor on every client/server."""
    torch.manual_seed(parameters_federated.EMBEDDING_EXTRACTOR_SEED)
    model = EmbeddingMLP(
        hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
        embedding_dim=parameters_federated.EMBEDDING_DIM,
    )

    path = getattr(parameters_federated, "EMBEDDING_EXTRACTOR_PATH", None)
    if path:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int):
    global fds
    if fds is None:
        if parameters_federated.PARTITIONER == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        else:
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=parameters_federated.DIRICHLET_ALPHA,
                min_partition_size=10,
                self_balancing=True,
                shuffle=True,
                seed=42,
            )
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2)

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((parameters_federated.MEAN,), (parameters_federated.STD,))]
    )

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(
        partition_train_test["train"],
        batch_size=parameters_federated.BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        partition_train_test["test"],
        batch_size=parameters_federated.BATCH_SIZE,
    )
    return train_loader, test_loader


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


# Original image-classifier training/testing kept for compatibility.
def train(net, train_loader, privacy_engine, optimizer, target_delta, device, epochs=1):
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
