from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

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

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

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
def resize_to_foundation_input(images, device, image_size=224):
    """Redimensiona e expande canais, mas MANTÉM a imagem em pixel space [0,1].

    Separado de `normalize_for_foundation` para permitir injetar ruído DEPOIS
    do resize (na resolução real de entrada do modelo), e não antes, o que
    antes deixava o upsample bilinear borrar/atenuar o ruído de alta
    frequência aplicado em 28x28.
    """
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
    return images


def normalize_for_foundation(images, device):
    """Aplica a normalização ImageNet. Espera imagem já redimensionada e em pixel space [0,1]."""
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - imagenet_mean) / imagenet_std


def preprocess_image(images, device, image_size=224):
    """Mantido por compatibilidade (usado quando não há ruído a injetar): resize + normalize."""
    images = resize_to_foundation_input(images, device, image_size=image_size)
    return normalize_for_foundation(images, device)


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
        generator = torch.Generator(device="cpu").manual_seed(image_noise_seed)

    embeddings_list = []
    labels_list = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        # 1) Redimensiona ANTES de aplicar ruído: assim o ruído passa a ser
        #    injetado na resolução real de entrada do modelo fundacional
        #    (image_size x image_size), e não mais em 28x28 seguido de
        #    upsample -- o que antes fazia a interpolação bilinear "borrar"
        #    parte do ruído de alta frequência antes dele chegar ao extrator.
        images = resize_to_foundation_input(
            images, device, image_size=parameters_federated.FOUNDATION_IMAGE_SIZE
        )

        if image_noise_std > 0:
            noise = torch.randn(
                images.shape,
                generator=generator,
                device="cpu",
                dtype=images.dtype,
            ).to(device)
            images = torch.clamp(images + image_noise_std * noise, 0.0, 1.0)

        # 2) Só então normaliza com as estatísticas do ImageNet.
        images = normalize_for_foundation(images, device)

        embeddings = foundation_model(images)
        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())

    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)


def build_or_load_embedding_loader(
    cache_path,
    foundation_model,
    image_loader,
    device,
    metadata,
    batch_size,
    image_noise_std=0.0,
    image_noise_seed=None,
):
    cache_path = Path(cache_path)

    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        if data.get("metadata", {}) == metadata:
            return make_embedding_loader(
                data["embeddings"],
                data["labels"],
                batch_size=batch_size,
                shuffle=False,
            )

    embeddings, labels = extract_embeddings_from_loader(
        foundation_model,
        image_loader,
        device,
        image_noise_std=image_noise_std,
        image_noise_seed=image_noise_seed,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings.cpu(),
            "labels": labels.cpu(),
            "metadata": metadata,
        },
        cache_path,
    )

    return make_embedding_loader(
        embeddings,
        labels,
        batch_size=batch_size,
        shuffle=False,
    )

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
        if parameters_federated.DATASET == "mnist":
            fds = FederatedDataset(
                dataset="ylecun/mnist",
                partitioners={"train": partitioner},
            )
        else:
            fds = FederatedDataset(
                dataset=f"danjacobellis/{parameters_federated.DATASET}_224",
                partitioners={"train": partitioner},
            )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

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