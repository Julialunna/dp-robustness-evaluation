from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset
import parameters_federated
import train


device = train.get_device()
#configuração dataset 
# MNIST = 28. MedMNIST tem variantes 28/64/128/22d4       
NUM_CLASSES = parameters_federated.NUM_CLASSES    
LATENT_DIM = parameters_federated.CVAE_LATENT_DIM  
       

#transforma rótulos inteiros em one-hot
def one_hot(labels, num_classes=NUM_CLASSES):
    return F.one_hot(labels.long(), num_classes).float().to(labels.device)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        #embedding concatenado com a classe, do mesmo jeito que é com CNN
        enc_in = input_dim + num_classes

        self.net = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, c):
        h = torch.cat([x, c], dim=1)

        h = self.net(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


# ----------------------------------------------------------------------
# Decoder convolucional: aproxima p_theta(x | z, c)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        dec_in = latent_dim + num_classes

        self.net = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, z, c):
        h = torch.cat([z, c], dim=1)
        return self.net(h)

#queremos amostrar z ~ N(mi, sigma^2), mas não podemos amostrar diretamente de uma distribuição normal e ainda assim ter um gradiente que flua de volta para mi e sigma. A reparametrização resolve isso: amostramos epsilon ~ N(0,1) e então transformamos z = mi + sigma * epsilon. Assim, o gradiente pode fluir através de mi e sigma.
#precisamos de uma função diferenciável, o problema é que z mudaria por acaso, aí não saberíamos se a imagem ficou ruim porque μ estava errado, porque σ estava errado, ou simplesmente porque o sorteio deu um z ruim? deixando a aleatoriedade apenas em epsilon coseguimeos diferenciar mi e sigma 

def reparameterize(mu, logvar):
    """z = mu + sigma * epsilon, epsilon ~ N(0,I) — truque de
    reparametrização que torna a amostragem diferenciável."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class CVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

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

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar

    def decode(self, z, c):
        return self.decoder(z, c)

#mudança: Opacus precisa saber qual a contribuição de cada exemplo para a loss, para fazer o clipping, então é melhor calcularmos a loss por exemplo e depois somar, ao invés de calcular a loss de uma vez só com reduction="sum"
def loss_function(x_hat, x, mu, logvar, beta=parameters_federated.CVAE_BETA):
    #binary_cross_entropy não faz mais sentido porque antes eram imagens com pixels entre 0 e 1, mas agora são embeddings que podem ter valores negativos. Então vamos usar MSE que mede a distância entre dois vetores contínuos 
    recon = F.mse_loss(
        x_hat, x, reduction="none"
    ).sum(dim=1)
    #dim = 1 ele calcula a loss dimensão por dimensão e então soma 

    #KLD novamente para forçar a distribuição latente a se aproximar de uma normal padrão 
    kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1)
    #beta alto espaço latente mais organizado mas a recontrução pode ser pior
    #beta mais baixo espaço latente mais bagunçado mas a reconstrução pode ser melhor
    loss_per_sample = recon + beta * kld_per_sample
    #assim temos um escalar para usar em loss.backward() 
    loss = loss_per_sample.mean()
    recon = recon.mean()
    #kld = kld_per_sample.mean()
    return loss

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
    input_dim: int,
    num_classes: int,
    latent_dim: int,
    hidden_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    beta: float,
    use_dp: bool,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    device: torch.device,
) -> DPCVAETrainResult:

    dataset = TensorDataset(embeddings.float(), labels.long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = CVAE( input_dim=input_dim,
    num_classes=num_classes,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,).to(device)
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
            c_batch = one_hot(y_batch, num_classes= num_classes)
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

    # If Opacus wrapped the module, return the underlying plain model for generation/saving.
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
    num_classes: int,
    latent_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.to(device)
    model.eval()

    labels = real_labels.long().to(device)
    #evita de o dataset ficar ordenado por classe 
    labels = labels[torch.randperm(len(labels), device=device)]

    num_samples = len(labels)


    z = torch.randn(num_samples, latent_dim, device=device)
    c = one_hot(labels, num_classes=num_classes)
    synthetic = model.decode(z, c)
    return synthetic.cpu(), labels.cpu()


def save_synthetic_dataset(path: str | Path, embeddings: torch.Tensor, labels: torch.Tensor, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings.cpu(), "labels": labels.cpu(), "metadata": metadata}, path)


def load_synthetic_dataset(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    data = torch.load(path, map_location="cpu")
    return data["embeddings"], data["labels"], data.get("metadata", {})
