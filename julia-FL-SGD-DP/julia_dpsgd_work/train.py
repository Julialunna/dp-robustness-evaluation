from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

import parameters_federated
import torch.nn.functional as F


from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

fds = None


def define_foundation_extractor(model_name: str, device: torch.device):
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

#o dinov2 faz center crop, corta para ficar só o centro da imagem, normalmente no ImageNet o objeto está no centro, mas no MEDMNIST pode não estar então não vou fazer center crop, vou só redimensionar para 224x224 e normalizar com os valores do ImageNet
def preprocess_image(images, device, image_size = 224):
    images = images.to(device).float()
    if images.dim() == 3:
        images = images.unsqueeze(1)
    #MNIST tem 1 canal, mas EfficientNet e MobileNet esperam 3 canais, então faço ter 3 
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    #aumenta o tamanho da imagem para 224x224, que é o tamanho esperado 
    images = F.interpolate(
        images,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    return (images - imagenet_mean) / imagenet_std

@torch.no_grad()
def extract_embeddings_from_loader(
    foundation_model,
    loader,
    device,
    image_noise_std=0.0,
    image_noise_seed=None,
):
    foundation_model.to(device)
    foundation_model.eval()

    generator = None
    if image_noise_seed is not None:
        #gerador de números aleatórios com seed fixa
        generator = torch.Generator(device=device).manual_seed(image_noise_seed)

    embeddings_list = []
    labels_list = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        if image_noise_std > 0:
            noise = torch.randn(
                images.shape,
                generator=generator,
                device=device,
                dtype=images.dtype,
            )
            images = torch.clamp(images + image_noise_std * noise, 0.0, 1.0)

        images = preprocess_image(
            images,
            device,
            image_size=parameters_federated.FOUNDATION_IMAGE_SIZE,
        )

        embeddings = foundation_model(images)
        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())

    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)

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
        if parameters_federated.PARTITIONER == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        elif parameters_federated.PARTITIONER == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                alpha=parameters_federated.DIRICHLET_ALPHA,
                min_partition_size=10,
                self_balancing=True,
                shuffle=True,
            )
        else:
            raise ValueError(f"Particionador desconhecido: {parameters_federated.PARTITIONER}")
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


def make_embedding_loader(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = torch.utils.data.TensorDataset(embeddings.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(net, embedding_loader,  optimizer, target_delta, device, privacy_engine = None,epochs=1):
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()

    for _ in range(epochs):
        for embeddings, labels in embedding_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            criterion(net(embeddings), labels).backward()
            optimizer.step()

    if privacy_engine is not None:
        epsilon = privacy_engine.get_epsilon(delta=target_delta)
        return epsilon

def test(net, embedding_loader, device):
    net.to(device)
    net.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    total = 0

    with torch.no_grad():
        for embeddings, labels in embedding_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = net(embeddings)
            loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / max(total, 1)
    loss = loss / max(total, 1)

    return loss, accuracy

@torch.no_grad()
def test_embedding_classifier_from_images(classifier, foundation_model, image_loader, device):
    embeddings, labels = extract_embeddings_from_loader(
        foundation_model,
        image_loader,
        device,
    )

    embedding_loader = make_embedding_loader(
        embeddings,
        labels,
        batch_size=parameters_federated.BATCH_SIZE,
        shuffle=False,
    )

    return test(classifier, embedding_loader, device)