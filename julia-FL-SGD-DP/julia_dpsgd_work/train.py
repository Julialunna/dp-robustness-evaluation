from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner

import parameters_federated
import utils 
import torch.nn.functional as F
from torchvision.models import resnet18
from datasets import DatasetDict

fds = None

# pretrain_mean, pretrain_std = utils.calculate_mean_std(
#     pretrain_loader
# )

# trained_resnet = train_embedding_extractor(
#     train_loader=pretrain_loader,
#     device=device,
#     mean=pretrain_mean,
#     std=pretrain_std,
# )

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def define_resnet18_classifier(
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def normalize_for_pretrained_resnet(
    images,
    mean,
    std,
    device,
):

    mean_tensor = torch.tensor(
        mean,
        device=device,
        dtype=images.dtype,
    )

    std_tensor = torch.tensor(
        std,
        device=device,
        dtype=images.dtype,
    )

    if mean_tensor.numel() == 1 and images.size(1) == 3:
        mean_tensor = mean_tensor.repeat(3)
        std_tensor = std_tensor.repeat(3)

    mean_tensor = mean_tensor.view(1, -1, 1, 1)
    std_tensor = std_tensor.view(1, -1, 1, 1)

    return (images - mean_tensor) / std_tensor

def train_embedding_extractor(train_loader, mean, std, device, epochs=parameters_federated.PRE_TRAIN_EPOCHS, lr=parameters_federated.PRE_TRAIN_LR):
    model = define_resnet18_classifier(
        num_classes=parameters_federated.NUM_CLASSES,
        device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            images = batch["image"].to(
                device=device,
                dtype=torch.float32,
            )

            labels = batch["label"]

            if not torch.is_tensor(labels):
                labels = torch.tensor(labels)

            labels = labels.to(
                device=device,
                dtype=torch.long,
            ).view(-1)

            if images.dim() == 3:
                images = images.unsqueeze(1)

            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)

            images = normalize_for_pretrained_resnet(
                images,
                mean,
                std,
                device,
            )

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_accuracy = total_correct / max(total_samples, 1)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={epoch_loss:.4f} | "
            f"accuracy={epoch_accuracy:.4f}"
        )

    model.eval()

    return model

def prepare_foundation_extractor(model_name: str, device: torch.device):
    if model_name == "dinov2_s":
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
def prepare_input_for_dino_vs2(images, device, image_size=224):
    images = images.to(device).float()
    if images.dim() == 3:
        images = images.unsqueeze(1)
    #MNIST tem 1 canal, mas EfficientNet e MobileNet esperam 3 canais, então faço ter 3 
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
        
    if images.shape[-2:] != (224, 224):
        raise ValueError(
            f"dinov2 espera imagens 224x224, "
            f"mas recebeu {tuple(images.shape[-2:])}"
        )
    return images


def normalize_for_foundation(images, device):
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images - imagenet_mean) / imagenet_std


def preprocess_image(images, device, image_size=224):
    images = prepare_input_for_dino_vs2(images, device, image_size=image_size)
    return normalize_for_foundation(images, device)


@torch.no_grad()
def extract_embeddings_from_loader(
    loader,
    device,
    image_noise_std=0.0,
    image_noise_seed=None,
):
    if parameters_federated.PRE_TRAIN_OR_EMBEDDING_MODEL == "dinov2_s":
        foundation_model = prepare_foundation_extractor(
            parameters_federated.FOUNDATION_MODEL,
            device,
        )
        foundation_model.to(device)
        foundation_model.eval()

    generator = None
    if image_noise_seed is not None:
        #gerador de números aleatórios com seed fixa
        generator = torch.Generator(device="cpu").manual_seed(image_noise_seed)

    embeddings_list = []
    labels_list = []
    if parameters_federated.PRE_TRAIN_OR_EMBEDDING_MODEL == "dinov2_s":
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            
          
            images = prepare_input_for_dino_vs2(
                    images, device, image_size=parameters_federated.FOUNDATION_IMAGE_SIZE)

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
            if isinstance(labels, (list, tuple)):
                if len(labels) == 1 and torch.is_tensor(labels[0]):
                    labels = labels[0]
                elif all(torch.is_tensor(label) for label in labels):
                    labels = torch.stack(labels, dim=-1)
                else:
                    labels = torch.tensor(labels)

            labels = labels.to(dtype=torch.long).reshape(-1)
            labels_list.append(labels.cpu())
    elif parameters_federated.PRE_TRAIN_OR_EMBEDDING_MODEL == "pre_train":
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            embeddings_list.append(images.cpu())
            if isinstance(labels, (list, tuple)):
                if len(labels) == 1 and torch.is_tensor(labels[0]):
                    labels = labels[0]
                elif all(torch.is_tensor(label) for label in labels):
                    labels = torch.stack(labels, dim=-1)
                else:
                    labels = torch.tensor(labels)

            labels = labels.to(dtype=torch.long).reshape(-1)
            labels_list.append(labels.cpu())

    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)


def build_or_load_embedding_loader(
    cache_path,
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

#para pegar o pre-train pretrain_dataset = fds.load_split("pretrain")    
def preprocess_dataset(dataset):
    def flatten_labels(batch):
        new_labels = []

        for label in batch["label"]:
            if isinstance(label, list):
                new_labels.append(int(label[0]))
            else:
                new_labels.append(int(label))

        batch["label"] = new_labels
        return batch

    dataset = dataset.map(
        flatten_labels,
        batched=True,
    )

    train_dataset = dataset["train"].class_encode_column("label")

    split = train_dataset.train_test_split(
        test_size=parameters_federated.PRETRAIN_DATA_FRACTION,
        seed=parameters_federated.DATA_SPLIT_SEED,
        shuffle=True,
        stratify_by_column="label",
    )

    processed_dataset = DatasetDict(
        {
            split_name: split_dataset
            for split_name, split_dataset in dataset.items()
            if split_name != "train"
        }
    )

    processed_dataset["train"] = split["train"]
    processed_dataset["pretrain"] = split["test"]

    return processed_dataset


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
                preprocessor=preprocess_dataset,
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