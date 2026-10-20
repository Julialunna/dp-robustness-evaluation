"""
Módulo de Modelos Downstream: Classificador de Embeddings e Métricas Avançadas.

Calcula Acurácia, Acurácia Balanceada e Macro F1-Score para avaliação sob distribuições não-IID.
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config


class EmbeddingClassifier(nn.Module):
    """Classificador MLP raso que opera sobre os vetores de embedding."""

    def __init__(
        self,
        input_dim: int = config.EMBEDDING_DIM,
        hidden_dim: int = config.CLASSIFIER_HIDDEN_DIM,
        num_classes: int = config.NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extrai os pesos de um modelo PyTorch como lista de arrays NumPy (para o Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Aplica uma lista de pesos NumPy em um modelo PyTorch."""
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({key: torch.tensor(val) for key, val in params_dict})
    model.load_state_dict(state_dict, strict=True)


def make_embedding_loader(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """Cria um DataLoader PyTorch a partir de tensores de embeddings e rótulos."""
    dataset = TensorDataset(embeddings.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = config.NUM_CLASSES,
) -> Dict[str, float]:
    """
    Calcula Acurácia Geral, Acurácia Balanceada (macro recall) e Macro F1-Score.
    
    Implementação vetorizada robusta para cenários Não-IID e classes desbalanceadas.
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0}

    accuracy = float(np.mean(y_true == y_pred))

    # Matriz de Confusão [num_classes, num_classes] (linhas: verdade, colunas: predito)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    recalls = []
    f1_scores = []

    for c in range(num_classes):
        tp = confusion[c, c]
        fn = np.sum(confusion[c, :]) - tp
        fp = np.sum(confusion[:, c]) - tp

        # Recall (sensibilidade por classe)
        class_support = tp + fn
        if class_support > 0:
            rec = tp / class_support
            recalls.append(rec)
        else:
            rec = 0.0

        # Precision por classe
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # F1 por classe
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0

        if class_support > 0:
            f1_scores.append(f1)

    balanced_accuracy = float(np.mean(recalls)) if recalls else 0.0
    f1_macro = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1_macro": f1_macro,
    }


def train_classifier(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = config.LOCAL_EPOCHS,
) -> float:
    """Executa o loop de treinamento supervisionado local do classificador."""
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()

    final_loss = 0.0
    for _ in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_n = labels.size(0)
            running_loss += loss.item() * batch_n
            total_samples += batch_n

        final_loss = running_loss / max(total_samples, 1)

    return final_loss


def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = config.NUM_CLASSES,
) -> Tuple[float, Dict[str, float]]:
    """
    Avalia a loss, acurácia, acurácia balanceada e macro F1-score do classificador.
    Retorna: (avg_loss, metrics_dict)
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = model(embeddings)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    if total > 0:
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        metrics = calculate_metrics(y_true, y_pred, num_classes=num_classes)
    else:
        metrics = {"accuracy": 0.0, "balanced_accuracy": 0.0, "f1_macro": 0.0}

    return avg_loss, metrics