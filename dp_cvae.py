"""
Módulo DP-CVAE: Conditional Variational Autoencoder com Privacidade Diferencial.

Utilizado pelos clientes federados para aprender a distribuição dos embeddings locais
e gerar dados sintéticos com garantias de privacidade (DP-SGD via Opacus).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

import config


def one_hot(labels: torch.Tensor, num_classes: int = config.NUM_CLASSES) -> torch.Tensor:
    """Converte rótulos inteiros de classe em vetores one-hot float."""
    return F.one_hot(labels.long(), num_classes).float().to(labels.device)


class Encoder(nn.Module):
    """Codificador do CVAE: Mapeia embedding x e rótulo c para a média (mu) e log-variância (logvar)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = config.NUM_CLASSES,
        latent_dim: int = config.CVAE_LATENT_DIM,
        hidden_dim: int = config.CVAE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        enc_in = input_dim + num_classes

        self.net = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([x, c], dim=1)
        h = self.net(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """Decodificador do CVAE: Reconstrói o embedding a partir do vetor latente z e rótulo c."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = config.NUM_CLASSES,
        latent_dim: int = config.CVAE_LATENT_DIM,
        hidden_dim: int = config.CVAE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        dec_in = latent_dim + num_classes
        reduced_dim = max(hidden_dim // 2, latent_dim)

        self.net = nn.Sequential(
            nn.Linear(dec_in, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, c], dim=1)
        return self.net(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Truque da reparametrização: z = mu + sigma * epsilon, epsilon ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class CVAE(nn.Module):
    """Conditional Variational Autoencoder completo."""

    def __init__(
        self,
        input_dim: int = config.EMBEDDING_DIM,
        num_classes: int = config.NUM_CLASSES,
        latent_dim: int = config.CVAE_LATENT_DIM,
        hidden_dim: int = config.CVAE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.decoder = Decoder(
            input_dim=input_dim,
            num_classes=num_classes,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, c)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, c)


def loss_function(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = config.CVAE_BETA,
) -> torch.Tensor:
    """
    Função de perda do CVAE = Erro Quadrático Médio (MSE) de Reconstrução + Beta * Divergência KL.
    Calculada por amostra para compatibilidade com o clipping de gradiente individual do Opacus.
    """
    recon_per_sample = F.mse_loss(x_hat, x, reduction="none").sum(dim=1)
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    loss_per_sample = recon_per_sample + beta * kld_per_sample
    return loss_per_sample.mean()


@dataclass
class DPCVAETrainResult:
    model: CVAE
    epsilon: Optional[float]
    num_examples: int
    final_loss: float


def train_local_dp_cvae(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    input_dim: int = config.EMBEDDING_DIM,
    num_classes: int = config.NUM_CLASSES,
    latent_dim: int = config.CVAE_LATENT_DIM,
    hidden_dim: int = config.CVAE_HIDDEN_DIM,
    batch_size: int = config.CVAE_BATCH_SIZE,
    epochs: int = config.CVAE_EPOCHS,
    lr: float = config.CVAE_LR,
    beta: float = config.CVAE_BETA,
    use_dp: bool = config.USE_LOCAL_DP_CVAE,
    target_epsilon: float = config.TARGET_EPSILON,
    target_delta: float = config.TARGET_DELTA,
    max_grad_norm: float = config.MAX_GRAD_NORM,
    device: torch.device,
) -> DPCVAETrainResult:
    """Treina o CVAE localmente nos embeddings reais com ou sem DP-SGD (Opacus)."""
    dataset = TensorDataset(embeddings.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CVAE(
        input_dim=input_dim,
        num_classes=num_classes,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    privacy_engine: Optional[PrivacyEngine] = None

    if use_dp:
        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

    final_loss = 0.0
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        running_examples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            c_batch = one_hot(y_batch, num_classes=num_classes)
            recon, mu, logvar = model(x_batch, c_batch)

            loss = loss_function(recon, x_batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            batch_n = y_batch.size(0)
            running_loss += loss.item() * batch_n
            running_examples += batch_n

        final_loss = running_loss / max(running_examples, 1)

    epsilon = None
    if privacy_engine is not None:
        epsilon = float(privacy_engine.get_epsilon(delta=target_delta))

    plain_model = model._module if hasattr(model, "_module") else model
    plain_model.eval()

    return DPCVAETrainResult(
        model=plain_model,
        epsilon=epsilon,
        num_examples=len(dataset),
        final_loss=float(final_loss),
    )


@torch.no_grad()
def generate_synthetic_embeddings(
    model: CVAE,
    *,
    real_labels: torch.Tensor,
    num_classes: int = config.NUM_CLASSES,
    latent_dim: int = config.CVAE_LATENT_DIM,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gera um conjunto de embeddings sintéticos condicionados à distribuição dos rótulos reais."""
    model.to(device)
    model.eval()

    labels = real_labels.long().to(device)
    labels = labels[torch.randperm(len(labels), device=device)]
    num_samples = len(labels)

    z = torch.randn(num_samples, latent_dim, device=device)
    c = one_hot(labels, num_classes=num_classes)
    synthetic = model.decode(z, c)

    return synthetic.cpu(), labels.cpu()


def save_synthetic_dataset(
    path: str | Path,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    metadata: dict,
) -> None:
    """Salva os embeddings sintéticos e metadados no disco."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings.cpu(),
            "labels": labels.cpu(),
            "metadata": metadata,
        },
        path,
    )


def load_synthetic_dataset(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Carrega embeddings sintéticos e metadados salvos no disco."""
    data = torch.load(path, map_location="cpu")
    return data["embeddings"], data["labels"], data.get("metadata", {})
