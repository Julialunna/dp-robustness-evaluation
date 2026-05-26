"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
     

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner,  DirichletPartitioner

import math
from tqdm import tqdm

import parameters_federated

fds = None  

class Net(nn.Module):

    def __init__(self, input_size=28*28, hidden_size=256, num_classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        y = self.fc1(x)
        y = self.relu1(y)
        y = self.fc2(y)
        y = self.relu2(y)
        y = self.fc3(y)

        return y
    
    
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
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)

    local_dp = AddLocalDP(parameters_federated.TARGET_EPSILON, parameters_federated.TARGET_DELTA, 784)

    pytorch_transforms = Compose([ToTensor(), local_dp, Normalize((parameters_federated.MEAN,), (parameters_federated.STD,))])



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

def train(net, train_loader, optimizer, device, epochs=1):
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



# class AddLocalDP:
#     def __init__(self, epsilon, delta, img_size):
#         sensibilidade = math.sqrt(img_size)
#         self.std = (sensibilidade * math.sqrt(2 * math.log(1.25 / delta))) / epsilon

#     def __call__(self, img):
#         noise = torch.randn_like(img) * self.std
#         return torch.clamp(img + noise, 0.0, 1.0)
    
from autodp.calibrator_zoo import eps_delta_calibrator
from autodp.mechanism_zoo import GaussianMechanism
class AddLocalDP:
    def __init__(self, epsilon, delta, img_size):
        sensibilidade = math.sqrt(img_size)
        calibrator = eps_delta_calibrator()
        mech = calibrator(GaussianMechanism, epsilon, delta, [0.001, 1000])
        self.std = mech.params['sigma'] * sensibilidade

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0.0, 1.0)