from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

import parameters_federated


from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

fds = None


def build_foundation_extractor(model_name: str, device: torch.device):
    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)

        extractor = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
        )

        embedding_dim = 1280

    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)

        extractor = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
        )

        embedding_dim = 576

    elif model_name == "dinov2_s":
        extractor = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
        )

        embedding_dim = 384

    else:
        raise ValueError(f"Modelo fundacional desconhecido: {model_name}")

    extractor.to(device)
    extractor.eval()

    for param in extractor.parameters():
        param.requires_grad = False

    return extractor, embedding_dim

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE, num_classes=10, dropout=0.2):
        super(EmbeddingClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

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
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
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

    pytorch_transforms = Compose([
        ToTensor()
    ])

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
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return loss, accuracy