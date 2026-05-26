"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from  torchvision.models import resnet18        

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, PathologicalPartitioner
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from tqdm import tqdm

import parameters_federated

fds = None  

class Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),         
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),               
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
# Rede que fiz na mao
# class Net(nn.Module):
#     def __init__(self, num_classes: int) -> None:
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size = x.size(0)
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(batch_size, -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)



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
        # partitioner = IidPartitioner(num_partitions=num_partitions)  # Particiona em IID
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            num_classes_per_partition=4,  # CLASSES POR NÓ. BEM HETEROGENEO
            partition_by="label",
        )
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    pytorch_transforms = Compose([ToTensor(), Normalize((parameters_federated.MEAN,), (parameters_federated.STD,))])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(
        partition_train_test["train"], batch_size=parameters_federated.BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(partition_train_test["test"], batch_size=parameters_federated.BATCH_SIZE)
    return train_loader, test_loader

# load_data(0, 100)

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
        # with BatchMemoryManager(
        #     data_loader=train_loader,
        #     max_physical_batch_size=32,
        #     optimizer=optimizer,
        # ) as memory_safe_data_loader:

        #     for batch in memory_safe_data_loader:

        #         images = batch["image"].to(device)
        #         labels = batch["label"].to(device)

        #         optimizer.zero_grad()

        #         outputs = net(images)

        #         loss = criterion(outputs, labels)

        #         loss.backward()

        #         optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return epsilon


def test(net, test_loader, device):
    net.to(device)
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