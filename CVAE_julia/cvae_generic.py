

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42)

#configuração dataset 
# MNIST = 1 (grayscale). PathMNIST/DermaMNIST/BloodMNIST = 3 (RGB)
IN_CHANNELS = 1      
# MNIST = 28. MedMNIST tem variantes 28/64/128/224
IMG_SIZE = 28         
NUM_CLASSES = 10     
LATENT_DIM = 32        

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
#numero de filtros de cada camada convolucional
BASE_CH = 32           
partitioner = DirichletPartitioner(
                num_partitions=10,
                partition_by="label",
                alpha=0.3,
                min_partition_size=10,
                self_balancing=True,
                shuffle=True,
                seed=42,)
fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": partitioner},
        )

partition = fds.load_partition(0)
#baixando dados e criando dataloaders
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


#transforma rótulos inteiros em one-hot
def one_hot(labels, num_classes=NUM_CLASSES):
    return F.one_hot(labels, num_classes).float().to(device)


#cada canal é uma matriz, MNIST como é gray scale tem 1, MEDMMINST com é colorido tem 3 (RGB) cada canal é uma matriz de 28x28, uma representa vermelho, outra verde e outra azul
#aqui, como temos 10 classes no MNIST, criamos 10 canais extras (.expand(-1, -1, h, w)), cada um representando uma classe, o canal da classe do exemplo é preenchido com 1 e os demais com 0, depois concatenamos com a imagem original (torch.cat([x, c_map], dim=1)), assim a rede convolucional vê a imagem e a classe ao mesmo tempo.
def broadcast_condition(c, h, w):
    return c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)


#padding: O padding é a técnica de adicionar uma "borda" de pixels extras (quase sempre preenchidos com o valor zero, o chamado Zero-Padding) ao redor da imagem original, antes do filtro começar a deslizar, para não "vermos" eles no filtro apenas uma vez 
class Encoder(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
                 latent_dim=LATENT_DIM, base_ch=BASE_CH):
        super().__init__()
        # +num_classes canais: condição concatenada espacialmente à imagem através do broadcast_condition
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, base_ch, 4, 2, 1),      # H/2
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            #nn.Conv2d(canais_entrada, canais_saida, kernel_size, stride, padding)
                # recebe 11 canais
                # produz 32 canais
                # usa filtro 4x4
                # anda de 2 em 2 pixels
                # usa padding 1
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),   
            #batch normalization: normaliza a saída da camada convolucional, estabilizando e acelerando o treinamento
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),                # H/8
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
        )
        #Essa camada força a saída da CNN a ter tamanho espacial 4x4.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(base_ch * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 4, latent_dim)

    def forward(self, x, c):
        c_map = broadcast_condition(c, x.size(2), x.size(3))
        h = torch.cat([x, c_map], dim=1)   # condicionamento de q_phi(z|x,c)
        h = self.conv(h)
        h = self.pool(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ----------------------------------------------------------------------
# Decoder convolucional: aproxima p_theta(x | z, c)
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
                 latent_dim=LATENT_DIM, base_ch=BASE_CH, img_size=IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.base_ch = base_ch
        self.fc = nn.Linear(latent_dim + num_classes, base_ch * 4 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1), 
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),      # 8 -> 16
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1),           # 16 -> 32
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_channels, 3, 1, 1),
        )

    def forward(self, z, c):
        h = torch.cat([z, c], dim=1)      
        h = self.fc(h)
        h = h.view(h.size(0), self.base_ch * 4, 4, 4)
        h = self.deconv(h)
        # interpolação final: garante saída exatamente no img_size alvo, no MNIST aqui teríamos 32x32 mas as imagens devem ser 28x28, então redimensionamos para 28x28. 
        h = F.interpolate(h, size=(self.img_size, self.img_size),
                           mode="bilinear", align_corners=False)
        return torch.sigmoid(h)

#queremos amostrar z ~ N(mi, sigma^2), mas não podemos amostrar diretamente de uma distribuição normal e ainda assim ter um gradiente que flua de volta para mi e sigma. A reparametrização resolve isso: amostramos epsilon ~ N(0,1) e então transformamos z = mi + sigma * epsilon. Assim, o gradiente pode fluir através de mi e sigma.
#precisamos de uma função diferenciável, o problema é que z mudaria por acaso, aí não saberíamos se a imagem ficou ruim porque μ estava errado, porque σ estava errado, ou simplesmente porque o sorteio deu um z ruim? deixando a aleatoriedade apenas em epsilon coseguimeos diferenciar mi e sigma 
def reparameterize(mu, logvar):
    """z = mu + sigma * epsilon, epsilon ~ N(0,I) — truque de
    reparametrização que torna a amostragem diferenciável."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = reparameterize(mu, logvar)
        x_hat = self.decoder(z, c)
        return x_hat, mu, logvar


def loss_function(x_hat, x, mu, logvar, beta=1.0):
    #x_hat-> imagem recosntruída 
    #view: x_hat = [batch, 1, 28, 28] -> x_hat = [batch, 784]
    recon = F.binary_cross_entropy(
        x_hat.view(x_hat.size(0), -1), x.view(x.size(0), -1), reduction="sum"
    )
    #KL verifica o quanto a distribuição q_phi(z|x,c) se afasta da distribuição p(z) = N(0,1). Quanto maior o KL, mais diferente é a distribuição aprendida da distribuição normal padrão
    #beta controla a importancia do kl 
    #Se o espaço latente aprendido não parecer uma normal padrão, sortear z aleatório pode cair em regiões vazias, onde o decoder nunca aprendeu a gerar nada bom.
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kld, recon, kld


def train():
    model = CVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    os.makedirs("samples", exist_ok=True)
    losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_recon, total_kld = 0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            if IN_CHANNELS == 1 and x.size(1) != 1:
                x = x.mean(dim=1, keepdim=True)   # salvaguarda caso o dataset venha RGB
            c = one_hot(y.to(device))

            optimizer.zero_grad()
            x_hat, mu, logvar = model(x, c)
            loss, recon, kld = loss_function(x_hat, x, mu, logvar, beta=1.0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
        n = len(train_loader.dataset)
        losses.append(total_loss / n)
        print(f"Epoch {epoch:02d} | loss={total_loss/n:.2f} "
              f"| recon={total_recon/n:.2f} | KL={total_kld/n:.2f}")
        
        if epoch == EPOCHS or epoch % 5 == 0:
            generate_samples(model, epoch)

    torch.save(model.state_dict(), "cvae_generic.pt")
    # Plot training losses
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    return model


@torch.no_grad()
def generate_samples(model, epoch, n_per_class=8):
    model.eval()
    z = torch.randn(n_per_class * NUM_CLASSES, LATENT_DIM).to(device)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(n_per_class).to(device)
    c = one_hot(labels)
    imgs = model.decoder(z, c)
    save_image(imgs, f"samples/epoch_{epoch:02d}.png", nrow=n_per_class)


if __name__ == "__main__":
    train()
