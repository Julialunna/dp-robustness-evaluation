"""cvae.py: estágio 1, um CVAE (autoencoder variacional condicional) que aprende p(embedding | classe).

Arquitetura e perda do dp_FL (dp_FL/dp_cvae.py), sem DP:
  encoder  [x, onehot(y)] -> 2 camadas ReLU de largura `oculta` -> (mu, logvar), de dimensão `latente`;
  decoder  [z, onehot(y)] -> max(oculta // 2, latente) -> oculta -> D;
  perda    MSE somada nas D dimensões + beta * KL, média no lote.
Diferenças em relação ao dp_FL:
  - os embeddings são padronizados (z-score por dimensão, com média e desvio do treino) antes do CVAE: no PathMNIST o
    desvio por dimensão vai de 0,7 a 2,9 e a média chega a 6 em módulo; sem isso, as dimensões de maior variância
    pesam mais na MSE e a rede recebe entradas descentradas. A geração devolve a escala original;
  - a época guardada é a de menor perda na validação.
Para gerar: z ~ N(0, I), escolha a classe y e decodifique (CVAE.gerar).
"""
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, d, c, latente, oculta):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + c, oculta), nn.ReLU(), nn.Linear(oculta, oculta), nn.ReLU())
        self.fc_mu = nn.Linear(oculta, latente)
        self.fc_logvar = nn.Linear(oculta, latente)

    def forward(self, x, c):
        h = self.net(torch.cat([x, c], 1))
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, d, c, latente, oculta):
        super().__init__()
        r = max(oculta // 2, latente)
        self.net = nn.Sequential(nn.Linear(latente + c, r), nn.ReLU(), nn.Linear(r, oculta), nn.ReLU(),
                                 nn.Linear(oculta, d))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], 1))


class CVAE(nn.Module):
    def __init__(self, d, n_classes, latente=32, oculta=256):
        super().__init__()
        self.d, self.n_classes, self.latente = d, n_classes, latente
        self.encoder = Encoder(d, n_classes, latente, oculta)
        self.decoder = Decoder(d, n_classes, latente, oculta)
        # média e desvio da padronização; ficam no state_dict, junto com os pesos
        self.register_buffer("media", torch.zeros(d))
        self.register_buffer("desvio", torch.ones(d))

    def onehot(self, y):
        return F.one_hot(y, self.n_classes).float()

    def forward(self, x, y):
        c = self.onehot(y)
        mu, logvar = self.encoder(x, c)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z, c), mu, logvar

    def padroniza(self, x):
        return (x - self.media) / self.desvio

    @torch.no_grad()
    def gerar(self, y, gerador=None, lote=8192):
        """Um embedding sintético por rótulo em y (tensor long), na escala ORIGINAL do DINOv2."""
        dev = self.media.device
        out = []
        for i in range(0, len(y), lote):
            yy = y[i:i + lote].to(dev)
            z = torch.randn(len(yy), self.latente, generator=gerador).to(dev)
            out.append((self.decoder(z, self.onehot(yy)) * self.desvio + self.media).cpu())
        return torch.cat(out)


def termos(x_hat, x, mu, logvar):
    """MSE somada nas dimensões e KL, por exemplo (vetores do tamanho do lote)."""
    rec = F.mse_loss(x_hat, x, reduction="none").sum(1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return rec, kl


@torch.no_grad()
def perda_validacao(model, X, y, beta, device, lote=4096):
    """Perda média (MSE + beta * KL) e R² da reconstrução pela média do posterior, no espaço padronizado."""
    model.eval()
    soma = sq = 0.0
    for i in range(0, len(X), lote):
        xb = model.padroniza(torch.as_tensor(X[i:i + lote], device=device))
        c = model.onehot(torch.as_tensor(y[i:i + lote], device=device))
        mu, logvar = model.encoder(xb, c)
        rec, kl = termos(model.decoder(mu + torch.randn_like(mu) * torch.exp(0.5 * logvar), c), xb, mu, logvar)
        soma += (rec + beta * kl).sum().item()
        sq += F.mse_loss(model.decoder(mu, c), xb, reduction="none").sum().item()
    var_total = model.padroniza(torch.as_tensor(X, device=device)).var(0, unbiased=False).sum().item()
    return soma / len(X), 1.0 - (sq / len(X)) / var_total


def treinar(Xtr, ytr, Xva, yva, n_classes, latente=32, oculta=256, epocas=30, lote=256, lr=1e-3, beta=1.0, seed=0,
            device="cpu", log=print):
    """Treina nos embeddings do treino; devolve (modelo da melhor época na validação, histórico, melhor época)."""
    torch.manual_seed(seed)
    model = CVAE(Xtr.shape[1], n_classes, latente, oculta).to(device)
    model.media.copy_(torch.as_tensor(Xtr.mean(0), device=device))
    model.desvio.copy_(torch.as_tensor(Xtr.std(0) + 1e-6, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device)
    g = torch.Generator().manual_seed(seed)
    hist, melhor, melhor_ep, estado = [], math.inf, 0, None
    for ep in range(1, epocas + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt), generator=g).to(device)
        soma = 0.0
        for i in range(0, len(perm), lote):
            idx = perm[i:i + lote]
            xb = model.padroniza(Xt[idx])
            x_hat, mu, logvar = model(xb, yt[idx])
            rec, kl = termos(x_hat, xb, mu, logvar)
            loss = (rec + beta * kl).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            soma += loss.item() * len(idx)
        val, r2 = perda_validacao(model, Xva, yva, beta, device)
        hist.append({"epoca": ep, "treino_perda": soma / len(perm), "val_perda": val, "val_r2": r2,
                     "segundos": time.time() - t0})
        if val < melhor:
            melhor, melhor_ep, estado = val, ep, copy.deepcopy(model.state_dict())
        log(f"  CVAE época {ep:3d}/{epocas}  treino {soma / len(perm):9.3f}  val {val:9.3f} (R² {r2:.3f})  "
            f"{time.time() - t0:5.1f} s")
    model.load_state_dict(estado)
    return model.eval(), hist, melhor_ep


def salvar(model, caminho, **extra):
    torch.save({"state_dict": model.state_dict(), "d": model.d, "n_classes": model.n_classes,
                "latente": model.latente, "oculta": model.encoder.net[0].out_features, **extra}, caminho)


def carregar(caminho, device):
    """Também lê o cvae_privado.pt do gerador de cvae_embeddings/ (mesmo formato)."""
    ck = torch.load(caminho, map_location=device, weights_only=False)
    model = CVAE(ck["d"], ck["n_classes"], ck["latente"], ck["oculta"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
