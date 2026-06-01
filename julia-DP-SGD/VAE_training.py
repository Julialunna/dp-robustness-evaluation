import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from datasets import load_dataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import matplotlib.pyplot as plt

import parameters_centralized


# ---------------------------------------------------------
# Configurações com fallback
# ---------------------------------------------------------
BATCH_SIZE = getattr(parameters_centralized, "BATCH_SIZE", 128)
EPOCHS = getattr(parameters_centralized, "EPOCHS", 30)
USE_DP = getattr(parameters_centralized, "USE_DP", False)
TARGET_EPSILON = getattr(parameters_centralized, "TARGET_EPSILON", 10.0)
TARGET_DELTA = getattr(parameters_centralized, "TARGET_DELTA", 1e-5)
MAX_GRAD_NORM = getattr(parameters_centralized, "MAX_GRAD_NORM", 1.0)

INPUT_DIM = 784
NUM_CLASSES = 10
HIDDEN_DIM = 512
LATENT_DIM = 8

# Comece com beta menor para não borrar demais.
# Depois você pode testar 0.1, 0.3, 0.5, 1.0.
BETA_KL = 0.1

SEED = 42
torch.manual_seed(SEED)


# ---------------------------------------------------------
# 1. Arquitetura do C-VAE
# ---------------------------------------------------------
class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder para MNIST.

    Encoder:
        imagem x + rótulo y -> mu, logvar

    Decoder:
        vetor latente z + rótulo y -> imagem reconstruída/gerada
    """

    def __init__(
        self,
        input_dim=784,
        hidden_dim=512,
        latent_dim=32,
        num_classes=10,
    ):
        super(CVAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder recebe imagem achatada + rótulo one-hot
        self.enc_fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder recebe z + rótulo one-hot
        self.dec_fc1 = nn.Linear(latent_dim + num_classes, hidden_dim // 2)
        self.dec_fc2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, y_one_hot):
        x = x.view(-1, self.input_dim)
        xy = torch.cat([x, y_one_hot], dim=1)

        h = F.relu(self.enc_fc1(xy))
        h = F.relu(self.enc_fc2(h))

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y_one_hot):
        zy = torch.cat([z, y_one_hot], dim=1)

        h = F.relu(self.dec_fc1(zy))
        h = F.relu(self.dec_fc2(h))

        # MNIST está em [0, 1], então usamos sigmoid.
        x_recon = torch.sigmoid(self.dec_out(h))

        return x_recon

    def forward(self, x, y_one_hot):
        mu, logvar = self.encode(x, y_one_hot)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y_one_hot)

        return recon_x, mu, logvar


# ---------------------------------------------------------
# 2. Função de perda do C-VAE
# ---------------------------------------------------------
def cvae_loss_function(recon_x, x, mu, logvar, beta=BETA_KL):
    """
    Loss = BCE + beta * KLD

    BCE: erro de reconstrução
    KLD: regularização do espaço latente

    beta menor ajuda no começo porque evita que o KL domine a reconstrução.
    """
    x_flat = x.view(-1, INPUT_DIM)

    bce = F.binary_cross_entropy(recon_x, x_flat, reduction="sum")

    kld = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    loss = (bce + beta * kld) / x.size(0)

    return loss, bce / x.size(0), kld / x.size(0)


# ---------------------------------------------------------
# 3. Preparação dos dados
# ---------------------------------------------------------
def load_mnist_for_cvae(batch_size):
    dataset = load_dataset("ylecun/mnist")

    # Para BCE, a imagem precisa ficar em [0, 1].
    pytorch_transforms = Compose([ToTensor()])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    train_data = dataset["train"].with_transform(apply_transforms)
    test_data = dataset["test"].with_transform(apply_transforms)

    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return trainloader, testloader


def labels_to_one_hot(labels, device):
    """
    Garante que labels esteja como tensor no device correto
    e converte para one-hot.
    """
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    labels = labels.long().to(device)
    y_one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).float()

    return labels, y_one_hot


def unwrap_opacus_model(model):
    """
    Quando usamos Opacus, o modelo pode ficar embrulhado em GradSampleModule.
    Para chamar métodos próprios, como decode(), usamos model._module.
    """
    if hasattr(model, "_module"):
        return model._module
    return model


# ---------------------------------------------------------
# 4. Visualizações
# ---------------------------------------------------------
def save_reconstruction_plot(model, testloader, device, filename="inferencia_dp_cvae.png"):
    raw_model = unwrap_opacus_model(model)
    raw_model.eval()

    batch = next(iter(testloader))

    images = batch["image"][:8].to(device)
    labels, y_one_hot = labels_to_one_hot(batch["label"][:8], device)

    with torch.no_grad():
        recon_images, _, _ = raw_model(images, y_one_hot)

    original_np = images.cpu().numpy()
    recon_np = recon_images.view(-1, 28, 28).cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    fig.suptitle("Comparação: Original vs Reconstrução pelo C-VAE")

    for i in range(8):
        axes[0, i].imshow(original_np[i][0], cmap="gray")
        axes[0, i].set_title(f"Original: {labels_np[i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon_np[i], cmap="gray")
        axes[1, i].set_title("Reconstruída")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Reconstruções salvas como: {filename}")
    plt.show()


def save_generation_plot(model, device, filename="teste_geracao_cvae.png"):
    """
    Gera um dígito de cada classe usando z aleatório + rótulo solicitado.
    """
    raw_model = unwrap_opacus_model(model)
    raw_model.eval()

    target_labels = torch.arange(NUM_CLASSES).to(device)
    y_one_hot = F.one_hot(target_labels, num_classes=NUM_CLASSES).float().to(device)

    #z = torch.randn(NUM_CLASSES, LATENT_DIM).to(device)
    z = 0.5 * torch.randn(NUM_CLASSES, LATENT_DIM).to(device)

    with torch.no_grad():
        generated = raw_model.decode(z, y_one_hot)

    generated_np = generated.view(-1, 28, 28).cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Dígitos Sintéticos (0-9) Gerados pelo C-VAE")

    for i in range(NUM_CLASSES):
        row = i // 5
        col = i % 5

        axes[row, col].imshow(generated_np[i], cmap="gray")
        axes[row, col].set_title(f"Rótulo: {i}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Geração salva como: {filename}")
    plt.show()


def save_generation_grid_per_class(
    model,
    device,
    samples_per_class=5,
    filename="grade_geracao_cvae.png",
):
    """
    Gera várias amostras por classe.
    Cada imagem é gerada a partir de:
        imagem_sintetica = Decoder(z, y)
    onde:
        z = ruído aleatório
        y = classe desejada (one-hot)
    """
    raw_model = unwrap_opacus_model(model)
    raw_model.eval()

    all_images = []
    all_info = []

    for digit in range(NUM_CLASSES):
        labels = torch.full(
            size=(samples_per_class,),
            fill_value=digit,
            dtype=torch.long,
            device=device,
        )

        y_one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).float()
        # z = torch.randn(samples_per_class, LATENT_DIM).to(device)
        z = 0.5 * torch.randn(samples_per_class, LATENT_DIM).to(device)

        with torch.no_grad():
            generated = raw_model.decode(z, y_one_hot)

        generated = generated.view(-1, 28, 28).cpu().numpy()

        for j in range(samples_per_class):
            all_images.append(generated[j])
            all_info.append((digit, j + 1))  # (classe, número da amostra)

    fig, axes = plt.subplots(NUM_CLASSES, samples_per_class, figsize=(samples_per_class * 2.2, NUM_CLASSES * 2.0))
    fig.suptitle("Várias amostras sintéticas por classe\n(cada imagem = Decoder(ruído z, classe y))", fontsize=14)

    idx = 0
    for row in range(NUM_CLASSES):
        for col in range(samples_per_class):
            classe, amostra = all_info[idx]
            axes[row, col].imshow(all_images[idx], cmap="gray")
            axes[row, col].axis("off")

            # título de cada imagem
            axes[row, col].set_title(f"c={classe}\na={amostra}", fontsize=8)

            # nome da classe na lateral esquerda
            if col == 0:
                axes[row, col].set_ylabel(f"Classe {classe}", fontsize=10, rotation=90, labelpad=10)

            idx += 1

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Grade de geração salva como: {filename}")
    plt.show()


# ---------------------------------------------------------
# 5. Treinamento centralizado com ou sem DP-SGD
# ---------------------------------------------------------
def run_cvae_centralized_dp(epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    trainloader, testloader = load_mnist_for_cvae(BATCH_SIZE)

    model = CVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
    ).to(device)

    model = ModuleValidator.fix(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    privacy_engine = None

    if USE_DP:
        print("Treinando C-VAE com DP-SGD via Opacus.")
        privacy_engine = PrivacyEngine(secure_mode=False)

        try:
            model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=trainloader,
                target_epsilon=TARGET_EPSILON,
                target_delta=TARGET_DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                grad_sample_mode="ew",
                epochs=epochs,
            )
        except TypeError:
            # Compatibilidade com versões do Opacus que não aceitam grad_sample_mode.
            model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=trainloader,
                target_epsilon=TARGET_EPSILON,
                target_delta=TARGET_DELTA,
                max_grad_norm=MAX_GRAD_NORM,
                epochs=epochs,
            )
    else:
        print("Treinando C-VAE sem DP. Use isso primeiro para depurar a qualidade da geração.")

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0
        total_examples = 0

        for batch in trainloader:
            images = batch["image"].to(device)
            labels, y_one_hot = labels_to_one_hot(batch["label"], device)

            optimizer.zero_grad()

            recon_images, mu, logvar = model(images, y_one_hot)

            # loss, bce, kld = cvae_loss_function(
            #     recon_images,
            #     images,
            #     mu,
            #     logvar,
            #     beta=BETA_KL,
            # )
            BETA_MAX = 0.3
            WARMUP_EPOCHS = 20

            beta_current = min(BETA_MAX, BETA_MAX * (epoch + 1) / WARMUP_EPOCHS)

            loss, bce, kld = cvae_loss_function(
                recon_images,
                images,
                mu,
                logvar,
                beta=beta_current,
            )
            
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_bce += bce.item() * batch_size
            total_kld += kld.item() * batch_size
            total_examples += batch_size

        avg_loss = total_loss / total_examples
        avg_bce = total_bce / total_examples
        avg_kld = total_kld / total_examples

        log_str = (
            f"Epoch: {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"BCE: {avg_bce:.4f} | "
            f"KLD: {avg_kld:.4f}"
        )
        log_str += f" | beta = {beta_current:.3f}"
        
        if USE_DP and privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=TARGET_DELTA)
            log_str += f" | ε = {epsilon:.2f}"

        print(log_str)

    print("Treinamento concluído.")

    raw_model = unwrap_opacus_model(model)
    torch.save(raw_model.state_dict(), "trained_cvae.pth")
    print("Pesos salvos como: trained_cvae.pth")

    save_reconstruction_plot(
        model=model,
        testloader=testloader,
        device=device,
        filename="inferencia_dp_cvae.png" if USE_DP else "inferencia_cvae_sem_dp.png",
    )

    save_generation_plot(
        model=model,
        device=device,
        filename="teste_geracao_cvae.png",
    )

    save_generation_grid_per_class(
        model=model,
        device=device,
        samples_per_class=5,
        filename="grade_geracao_cvae.png",
    )

    return model


# ---------------------------------------------------------
# 6. Execução principal
# ---------------------------------------------------------
if __name__ == "__main__":
    trained_model = run_cvae_centralized_dp(
        epochs=EPOCHS,
        lr=1e-3,
    )