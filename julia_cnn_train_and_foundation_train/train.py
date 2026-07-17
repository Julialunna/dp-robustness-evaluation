from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)
from torchvision.transforms import Compose, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

import parameters_federated

fds = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CNNEmbeddingExtractor(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embedding_dim=128,
        base_channels=32,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, embedding_dim),
            nn.Sigmoid(),
        )

        self._uses_imagenet_normalization = False

    def forward(self, images):
        features = self.features(images)
        return self.embedding_head(features)


class CNNPretrainClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_classes,
        base_channels,
    ):
        super().__init__()
        self.extractor = CNNEmbeddingExtractor(
            in_channels=3,
            embedding_dim=embedding_dim,
            base_channels=base_channels,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        embeddings = self.extractor(images)
        return self.classifier(embeddings)


def get_dataset_name():
    if parameters_federated.DATASET == "mnist":
        return "ylecun/mnist"
    return f"danjacobellis/{parameters_federated.DATASET}_224"


def get_cnn_checkpoint_path():
    return Path(parameters_federated.CNN_CHECKPOINT_PATH)


def _cnn_checkpoint_metadata():
    return {
        "dataset": parameters_federated.DATASET,
        "embedding_dim": parameters_federated.EMBEDDING_DIM,
        "base_channels": parameters_federated.CNN_BASE_CHANNELS,
        "image_size": parameters_federated.FOUNDATION_IMAGE_SIZE,
        "num_classes": parameters_federated.NUM_CLASSES,
        "pretrain_ratio": parameters_federated.PRETRAIN_RATIO,
        "data_split_seed": parameters_federated.DATA_SPLIT_SEED,
    }


def _checkpoint_matches(checkpoint):
    return checkpoint.get("metadata", {}) == _cnn_checkpoint_metadata()


def _apply_image_transforms(dataset):
    pytorch_transforms = Compose([ToTensor()])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
        return batch

    return dataset.with_transform(apply_transforms)


def _prepare_labels(labels):
    if isinstance(labels, (list, tuple)):
        if len(labels) == 1 and torch.is_tensor(labels[0]):
            labels = labels[0]
        elif all(torch.is_tensor(label) for label in labels):
            labels = torch.stack(labels, dim=-1)
        else:
            labels = torch.tensor(labels)

    labels = labels.to(dtype=torch.long).reshape(-1)
    return labels


def resize_to_foundation_input(images, device, image_size=224):
    images = images.to(device).float()

    if images.dim() == 3:
        images = images.unsqueeze(1)

    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)

    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    return images


def normalize_for_foundation(images, device):
    imagenet_mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device=device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    imagenet_std = torch.tensor(
        [0.229, 0.224, 0.225],
        device=device,
        dtype=images.dtype,
    ).view(1, 3, 1, 1)
    return (images - imagenet_mean) / imagenet_std


def prepare_images_for_extractor(images, extractor, device, image_size=224):
    images = resize_to_foundation_input(images, device, image_size=image_size)

    if getattr(extractor, "_uses_imagenet_normalization", True):
        images = normalize_for_foundation(images, device)

    return images


def preprocess_image(images, device, image_size=224, extractor=None):
    images = resize_to_foundation_input(images, device, image_size=image_size)

    if extractor is None or getattr(extractor, "_uses_imagenet_normalization", True):
        images = normalize_for_foundation(images, device)

    return images


def _run_pretrain_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        images = prepare_images_for_extractor(
            batch["image"],
            model.extractor,
            device,
            image_size=parameters_federated.FOUNDATION_IMAGE_SIZE,
        )
        labels = _prepare_labels(batch["label"]).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return (
        total_loss / max(total_examples, 1),
        total_correct / max(total_examples, 1),
    )


def _split_train_dataset(train_dataset):
    pretrain_ratio = float(parameters_federated.PRETRAIN_RATIO)
    if not 0.0 < pretrain_ratio < 1.0:
        raise ValueError("PRETRAIN_RATIO deve estar entre 0 e 1.")

    train_dataset = train_dataset.shuffle(seed=parameters_federated.DATA_SPLIT_SEED)
    num_public = int(round(len(train_dataset) * pretrain_ratio))
    public_dataset = train_dataset.select(range(num_public))
    private_dataset = train_dataset.select(range(num_public, len(train_dataset)))
    return public_dataset, private_dataset


def get_public_private_splits():
    dataset = load_dataset(get_dataset_name())
    train_dataset = preprocess_dataset(dataset["train"])
    return _split_train_dataset(train_dataset)


def pretrain_cnn_extractor(device=None):
    device = device or get_device()
    public_dataset, private_dataset = get_public_private_splits()
    pretrain_data = _apply_image_transforms(public_dataset)

    train_loader = DataLoader(
        pretrain_data,
        batch_size=parameters_federated.PRETRAIN_BATCH_SIZE,
        shuffle=True,
    )

    torch.manual_seed(parameters_federated.PRETRAIN_SEED)

    model = CNNPretrainClassifier(
        embedding_dim=parameters_federated.EMBEDDING_DIM,
        num_classes=parameters_federated.NUM_CLASSES,
        base_channels=parameters_federated.CNN_BASE_CHANNELS,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=parameters_federated.PRETRAIN_LR,
        weight_decay=parameters_federated.PRETRAIN_WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    print(
        f"[CNN] Pré-treinamento com {len(pretrain_data)} exemplos. "
        f"Os outros {len(private_dataset)} exemplos serão usados pelos clientes."
    )

    for epoch in range(parameters_federated.PRETRAIN_EPOCHS):
        train_loss, train_accuracy = _run_pretrain_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{parameters_federated.PRETRAIN_EPOCHS} | "
            f"loss={train_loss:.4f} | accuracy={train_accuracy:.4f}"
        )

    checkpoint_path = get_cnn_checkpoint_path()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.extractor.state_dict(),
            "metadata": _cnn_checkpoint_metadata(),
        },
        checkpoint_path,
    )

    print(f"[CNN] Extratora salva em {checkpoint_path}")


def ensure_embedding_extractor_ready():
    if parameters_federated.FOUNDATION_MODEL != "cnn":
        return

    checkpoint_path = get_cnn_checkpoint_path()

    if checkpoint_path.exists() and not parameters_federated.FORCE_RETRAIN_EXTRACTOR:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if _checkpoint_matches(checkpoint):
            print(f"[CNN] Usando extratora salva em {checkpoint_path}")
            return

        print("[CNN] Checkpoint incompatível com a configuração atual. Retreinando.")

    if not parameters_federated.PRETRAIN_EXTRACTOR:
        raise FileNotFoundError(
            f"A CNN não está salva em {checkpoint_path} e PRETRAIN_EXTRACTOR=False."
        )

    pretrain_cnn_extractor()


def define_foundation_extractor(model_name: str, device: torch.device):
    if model_name == "cnn":
        extractor = CNNEmbeddingExtractor(
            in_channels=3,
            embedding_dim=parameters_federated.EMBEDDING_DIM,
            base_channels=parameters_federated.CNN_BASE_CHANNELS,
        )
        checkpoint_path = get_cnn_checkpoint_path()

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint da CNN não encontrado em {checkpoint_path}. "
                "Execute o projeto pelo main.py para pré-treinar a extratora antes do Flower."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not _checkpoint_matches(checkpoint):
            raise ValueError(
                "O checkpoint da CNN não corresponde ao dataset ou à arquitetura atual."
            )

        extractor.load_state_dict(checkpoint["state_dict"], strict=True)
        embedding_dim = parameters_federated.EMBEDDING_DIM

    elif model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        extractor = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
        )
        extractor._uses_imagenet_normalization = True
        embedding_dim = 1280

    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        extractor = nn.Sequential(
            model.features,
            model.avgpool,
            nn.Flatten(),
        )
        extractor._uses_imagenet_normalization = True
        embedding_dim = 576

    elif model_name == "dinov2_s":
        extractor = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            trust_repo=True,
        )
        extractor._uses_imagenet_normalization = True
        embedding_dim = 384

    else:
        raise ValueError(f"Modelo extrator desconhecido: {model_name}")

    extractor.to(device)
    extractor.eval()

    for parameter in extractor.parameters():
        parameter.requires_grad = False

    return extractor, embedding_dim


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
        generator = torch.Generator(device="cpu").manual_seed(image_noise_seed)

    embeddings_list = []
    labels_list = []

    for batch in loader:
        images = resize_to_foundation_input(
            batch["image"],
            device,
            image_size=parameters_federated.FOUNDATION_IMAGE_SIZE,
        )

        if image_noise_std > 0:
            noise = torch.randn(
                images.shape,
                generator=generator,
                device="cpu",
                dtype=images.dtype,
            ).to(device)
            images = torch.clamp(images + image_noise_std * noise, 0.0, 1.0)

        if getattr(foundation_model, "_uses_imagenet_normalization", True):
            images = normalize_for_foundation(images, device)

        embeddings = foundation_model(images)
        labels = _prepare_labels(batch["label"])

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
    def __init__(
        self,
        input_size,
        hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
        num_classes=10,
        dropout=0.2,
    ):
        super().__init__()
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
        return self.net(x.view(x.size(0), -1))


def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def get_weights(net):
    return [value.cpu().numpy() for _, value in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({key: torch.tensor(value) for key, value in params_dict})
    net.load_state_dict(state_dict, strict=True)


def preprocess_dataset(dataset):
    def flatten_labels(batch):
        new_labels = []

        for label in batch["label"]:
            if isinstance(label, list):
                if len(label) != 1:
                    raise ValueError(
                        "Este pipeline exige um único rótulo por imagem. "
                        "ChestMNIST é multirrótulo e precisa de outra loss e outras métricas."
                    )
                new_labels.append(int(label[0]))
            else:
                new_labels.append(int(label))

        batch["label"] = new_labels
        return batch

    return dataset.map(flatten_labels, batched=True)


def preprocess_federated_dataset(dataset):
    dataset = preprocess_dataset(dataset)
    _, private_dataset = _split_train_dataset(dataset["train"])
    dataset["train"] = private_dataset
    return dataset


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
            raise ValueError(
                f"Particionador desconhecido: {parameters_federated.PARTITIONER}"
            )

        fds = FederatedDataset(
            dataset=get_dataset_name(),
            partitioners={"train": partitioner},
            preprocessor=preprocess_federated_dataset,
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    pytorch_transforms = Compose([ToTensor()])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(image) for image in batch["image"]]
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
):
    dataset = torch.utils.data.TensorDataset(embeddings.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(
    net,
    embedding_loader,
    optimizer,
    target_delta,
    device,
    privacy_engine=None,
    epochs=1,
):
    criterion = nn.CrossEntropyLoss()
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
        return privacy_engine.get_epsilon(delta=target_delta)


def test(net, embedding_loader, device):
    net.to(device)
    net.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    loss = 0.0
    total = 0

    with torch.no_grad():
        for embeddings, labels in embedding_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = net(embeddings)
            loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def test_embedding_classifier_from_images(
    classifier,
    foundation_model,
    image_loader,
    device,
):
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
