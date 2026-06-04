"""
Conditional Variational Autoencoder (CVAE) com DP-SGD via Opacus
Dataset: MNIST
Autor: gerado como base para aprendizado federado com privacidade diferencial

Estrutura:
  - Hiperparâmetros centralizados no topo
  - FLAG use_dp para ativar/desativar privacidade diferencial
  - Geração de imagens por classe ao final do treino
  - Métricas: ELBO (loss total), reconstruction loss, KL divergence, epsilon gasto
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# Opacus (DP-SGD)
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("[AVISO] Opacus não encontrado. Instale com: pip install opacus")
    print("        O treinamento continuará SEM privacidade diferencial.\n")


# ─────────────────────────────────────────────
# 1. HIPERPARÂMETROS  ← edite aqui
# ─────────────────────────────────────────────
CONFIG = {
    # ── Geral ──────────────────────────────
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "cvae_outputs",      # pasta onde figuras serão salvas

    # ── Dataset ────────────────────────────
    "data_root": "./data",
    "num_classes": 10,                 # dígitos 0-9

    # ── Arquitetura CVAE ───────────────────
    "input_dim": 784,                  # 28×28
    "hidden_dim": 512,                 # camadas intermediárias
    "latent_dim": 32,                  # dimensão do espaço latente z

    # ── Treinamento ────────────────────────
    "epochs": 30,
    "batch_size": 256,
    "lr": 1e-3,
    "beta_kl": 1.0,                    # peso da KL (beta-VAE; 1.0 = VAE padrão)

    # ── Privacidade Diferencial ────────────
    "use_dp": True,                    # False → desativa DP (modo teste)
    "target_epsilon": 10.0,            # orçamento de privacidade ε
    "target_delta": 1e-5,              # δ (tipicamente 1/N)
    "max_grad_norm": 1.0,              # clipping de gradiente por amostra

    # ── Geração de imagens ─────────────────
    "samples_per_class": 8,            # imagens geradas por dígito na grade final
}


# ─────────────────────────────────────────────
# 2. ARQUITETURA CVAE
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """
    q(z | x, y):  x (imagem achatada) + y (label one-hot) → μ, log σ²
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y_onehot):
        h = self.net(torch.cat([x, y_onehot], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """
    p(x | z, y):  z + y (label one-hot) → x̂ (reconstrução)
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),              # pixels em [0, 1]
        )

    def forward(self, z, y_onehot):
        return self.net(torch.cat([z, y_onehot], dim=-1))


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim  = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_classes)

    def reparameterize(self, mu, logvar):
        """Truque da reparametrização: z = μ + ε·σ,  ε ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        y = F.one_hot(labels, self.num_classes).float()
        mu, logvar = self.encoder(x, y)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decoder(z, y)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, labels, device):
        """Gera amostras condicionadas em `labels` (tensor de inteiros)."""
        y   = F.one_hot(labels, self.num_classes).float().to(device)
        z   = torch.randn(len(labels), self.latent_dim, device=device)
        return self.decoder(z, y)


# ─────────────────────────────────────────────
# 3. FUNÇÃO DE PERDA (ELBO)
# ─────────────────────────────────────────────

def cvae_loss(x, x_hat, mu, logvar, beta=1.0):
    """
    ELBO = E[log p(x|z,y)]  −  β · KL(q(z|x,y) ‖ p(z))
    Usamos BCE como proxy de log-verossimilhança.
    """
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total      = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ─────────────────────────────────────────────
# 4. TREINAMENTO
# ─────────────────────────────────────────────

def train_cvae(config: dict):
    # ── Reprodutibilidade ──────────────────
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device     = torch.device(config["device"])
    use_dp     = config["use_dp"] and OPACUS_AVAILABLE
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)
    print(f"  CVAE + {'DP-SGD (Opacus)' if use_dp else 'SGD sem DP'}")
    print(f"  Device : {device}")
    print(f"  Épocas : {config['epochs']}  |  Batch: {config['batch_size']}")
    if use_dp:
        print(f"  ε alvo : {config['target_epsilon']}  |  δ = {config['target_delta']}")
        print(f"  Clipping: {config['max_grad_norm']}")
    print("=" * 55)

    # ── Dataset ────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),   # achata 28×28 → 784
    ])
    train_dataset = datasets.MNIST(
        root=config["data_root"], train=True,
        download=True, transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,   # Opacus exige batches de mesmo tamanho
    )

    # ── Modelo ─────────────────────────────
    model = CVAE(
        input_dim   = config["input_dim"],
        hidden_dim  = config["hidden_dim"],
        latent_dim  = config["latent_dim"],
        num_classes = config["num_classes"],
    ).to(device)

    # Valida compatibilidade com Opacus (substitui BatchNorm, etc.)
    if use_dp:
        model = ModuleValidator.fix(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # ── Privacidade Diferencial ────────────
    privacy_engine = None
    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module        = model,
            optimizer     = optimizer,
            data_loader   = train_loader,
            epochs        = config["epochs"],
            target_epsilon= config["target_epsilon"],
            target_delta  = config["target_delta"],
            max_grad_norm = config["max_grad_norm"],
        )
        # Exibe o σ calculado
        print(f"  Noise multiplier (σ): {optimizer.noise_multiplier:.4f}\n")

    # ── Loop de treino ─────────────────────
    history = {"loss": [], "recon": [], "kl": [], "epsilon": []}

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = epoch_recon = epoch_kl = 0.0

        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, labels)
            loss, recon, kl   = cvae_loss(x, x_hat, mu, logvar, config["beta_kl"])
            loss.backward()
            optimizer.step()

            epoch_loss  += loss.item()
            epoch_recon += recon.item()
            epoch_kl    += kl.item()

        n_batches = len(train_loader)
        avg_loss  = epoch_loss  / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches

        history["loss"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        # ε gasto até agora
        eps_str = ""
        if use_dp and privacy_engine:
            eps = privacy_engine.get_epsilon(config["target_delta"])
            history["epsilon"].append(eps)
            eps_str = f"  ε = {eps:.4f}"

        print(
            f"Época [{epoch:>3}/{config['epochs']}]  "
            f"Loss: {avg_loss:.4f}  "
            f"Recon: {avg_recon:.4f}  "
            f"KL: {avg_kl:.4f}"
            f"{eps_str}"
        )

    # Epsilon final
    if use_dp and privacy_engine:
        final_eps = privacy_engine.get_epsilon(config["target_delta"])
        print(f"\n  ✓ Treinamento concluído  |  ε final = {final_eps:.4f}  (δ = {config['target_delta']})")
    else:
        print("\n  ✓ Treinamento concluído (sem DP)")

    return model, history


# ─────────────────────────────────────────────
# 5. GERAÇÃO DE IMAGENS POR CLASSE
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_and_plot(model, config: dict, history: dict):
    device         = torch.device(config["device"])
    num_classes    = config["num_classes"]
    n_samples      = config["samples_per_class"]
    output_dir     = config["output_dir"]

    model.eval()

    # ── Grade de amostras geradas ──────────
    fig, axes = plt.subplots(num_classes, n_samples, figsize=(n_samples * 1.4, num_classes * 1.4))
    dp_text = (
    f"+ DP (ε≈{config['target_epsilon']})"
    if config["use_dp"]
    else "(sem DP)"
    )

    fig.suptitle(
        f"CVAE {dp_text} — amostras geradas por classe",
        fontsize=11,
        y=1.01,
    )

    for digit in range(num_classes):
        labels  = torch.full((n_samples,), digit, dtype=torch.long)
        samples = model.sample(labels, device)           # (n, 784)
        samples = samples.cpu().view(n_samples, 28, 28).numpy()

        for j in range(n_samples):
            axes[digit, j].imshow(samples[j], cmap="gray", vmin=0, vmax=1)
            axes[digit, j].axis("off")
            if j == 0:
                axes[digit, j].set_ylabel(str(digit), rotation=0,
                                          labelpad=12, fontsize=10, va="center")

    plt.tight_layout()
    path = os.path.join(output_dir, "generated_digits.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Grade de imagens salva em: {path}")

    # ── Curvas de treino ───────────────────
    epochs = range(1, len(history["loss"]) + 1)

    n_plots = 3 + (1 if history["epsilon"] else 0)
    fig2, axs = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    axs[0].plot(epochs, history["loss"],  color="#4C72B0")
    axs[0].set_title("Loss Total (ELBO)"); axs[0].set_xlabel("Época")

    axs[1].plot(epochs, history["recon"], color="#DD8452")
    axs[1].set_title("Reconstruction Loss"); axs[1].set_xlabel("Época")

    axs[2].plot(epochs, history["kl"],    color="#55A868")
    axs[2].set_title("KL Divergence"); axs[2].set_xlabel("Época")

    if history["epsilon"]:
        axs[3].plot(epochs, history["epsilon"], color="#C44E52")
        axs[3].axhline(config["target_epsilon"], ls="--", color="gray", label=f"ε alvo = {config['target_epsilon']}")
        axs[3].set_title("ε acumulado (DP)"); axs[3].set_xlabel("Época")
        axs[3].legend(fontsize=8)

    for ax in axs:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path2, dpi=120)
    plt.show()
    print(f"  ✓ Curvas de treino salvas em: {path2}")


# ─────────────────────────────────────────────
# 6. SALVAR / CARREGAR MODELO
# ─────────────────────────────────────────────

def save_model(model, config: dict, path: str | None = None):
    """Salva pesos e configuração para uso posterior no aprendizado federado."""
    if path is None:
        tag  = f"dp_eps{config['target_epsilon']}" if config["use_dp"] else "no_dp"
        path = os.path.join(config["output_dir"], f"cvae_{tag}.pt")
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"  ✓ Modelo salvo em: {path}")
    return path


def load_model(path: str, device: str = "cpu") -> tuple[CVAE, dict]:
    """Carrega modelo e config de um checkpoint."""
    ckpt   = torch.load(path, map_location=device)
    config = ckpt["config"]
    model  = CVAE(
        input_dim   = config["input_dim"],
        hidden_dim  = config["hidden_dim"],
        latent_dim  = config["latent_dim"],
        num_classes = config["num_classes"],
    ).to(device)
    # Remove prefixo '_module.' adicionado pelo Opacus, se houver
    state = {k.replace("_module.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


# ─────────────────────────────────────────────
# 7. GERAÇÃO DE DATASET SINTÉTICO (para Federated Learning)
# ─────────────────────────────────────────────

@torch.no_grad()
def generate_synthetic_dataset(model, config: dict, n_per_class: int = 500):
    """
    Gera `n_per_class` imagens sintéticas por dígito.
    Retorna tensores (images, labels) prontos para uso em DataLoader.
    Útil para o cliente no aprendizado federado: cada cliente treina seu CVAE
    e depois gera dados sintéticos diferencialmente privados para o modelo global.
    """
    device  = torch.device(config["device"])
    model.eval()

    all_images, all_labels = [], []
    for digit in range(config["num_classes"]):
        labels  = torch.full((n_per_class,), digit, dtype=torch.long)
        samples = model.sample(labels, device).cpu()     # (n, 784)
        images  = samples.view(n_per_class, 1, 28, 28)   # → (n, 1, 28, 28)
        all_images.append(images)
        all_labels.append(labels)

    images_tensor = torch.cat(all_images, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    print(f"  ✓ Dataset sintético gerado: {images_tensor.shape[0]} amostras "
          f"({n_per_class} por classe)")
    return images_tensor, labels_tensor


# ─────────────────────────────────────────────
# 8. PONTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Treino ────────────────────────────
    model, history = train_cvae(CONFIG)

    # ── Geração e visualização ─────────────
    generate_and_plot(model, CONFIG, history)

    # ── Salvar modelo ─────────────────────
    save_model(model, CONFIG)

    # ── Exemplo: gerar dataset sintético ──
    images, labels = generate_synthetic_dataset(model, CONFIG, n_per_class=200)

    # ── Exemplo: recarregar modelo ─────────
    # tag   = f"dp_eps{CONFIG['target_epsilon']}" if CONFIG["use_dp"] else "no_dp"
    # path  = os.path.join(CONFIG["output_dir"], f"cvae_{tag}.pt")
    # model_loaded, cfg_loaded = load_model(path, device=CONFIG["device"])