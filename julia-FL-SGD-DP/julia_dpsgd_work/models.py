import torch
import torch.nn as nn
import torch.nn.functional as F
import parameters_federated

class EmbeddingClassifier(nn.Module):
    def __init__(
        self,
        hidden_size=parameters_federated.EMBEDDING_HIDDEN_SIZE,
        num_classes=10,
        dropout=0.2,
    ):
        super().__init__()
        EMBEDDING_DIMS = {"cnn": parameters_federated.EMBEDDING_DIM, "mlp": parameters_federated.EMBEDDING_DIM, "dinov2_s": 384}
        input_size = EMBEDDING_DIMS[parameters_federated.FOUNDATION_MODEL]
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
    
class MLPEmbeddingExtractor(nn.Module):
    def __init__(self, input_size, embedding_dim=128, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, images):
        x = images.view(images.size(0), -1)
        return self.net(x)


class MLPPretrainClassifier(nn.Module):
    def __init__(self, input_size, embedding_dim, num_classes, hidden_size):
        super().__init__()
        self.extractor = MLPEmbeddingExtractor(
            input_size=input_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images):
        embeddings = self.extractor(images)
        return self.classifier(embeddings)