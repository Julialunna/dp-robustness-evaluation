"""difusao.py: estágio 2, um modelo de difusão que aprende p(imagem | embedding DINOv2).

Por que difusão: um decodificador determinístico (embedding -> imagem treinado com perda de pixel) devolve a MÉDIA das
imagens compatíveis com o embedding e a imagem sai borrada, porque o embedding não fixa a textura fina. A difusão
AMOSTRA uma imagem compatível com o embedding: a morfologia sai nítida e plausível.

Modelo: U-Net (blocos residuais com GroupNorm; o tempo e a condição modulam cada bloco por escala e deslocamento;
autoatenção nas resoluções 16 e 8). Imagens em [-1, 1].
Condição: o embedding padronizado (z-score com média e desvio do treino), por um MLP; opcionalmente também a classe.
Em uma fração p_sem_cond dos exemplos de treino a condição vira um vetor "nulo" aprendido: isso permite a guia sem
classificador na amostragem, v = v_nulo + guia * (v_cond - v_nulo), que reforça a condição.
Difusão: tempo contínuo t em [0, 1], agenda cosseno (alfa = cos(pi t / 2), sigma = sin(pi t / 2)), a rede prevê
v = alfa * ruído - sigma * imagem; amostragem DDIM determinística em `passos` passos. Os pesos usados para gerar são a
média móvel exponencial (EMA) dos pesos de treino.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def autocast(device):
    """bfloat16 na GPU (mais rápido e cabe mais lote); precisão normal na CPU."""
    if str(device).startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def para_menos1_1(imgs_u8, device):
    """uint8 (n, H, W[, C]) -> tensor (n, C, H, W) em [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).to(device).float().div_(127.5).sub_(1)
    return x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)


def para_u8(x):
    """tensor (n, C, H, W) em [-1, 1] -> uint8 (n, H, W[, C])."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
    return (x[:, 0] if x.size(1) == 1 else x.permute(0, 2, 3, 1)).numpy()


# ---------------------------------------------------------------- a U-Net


def _gn(c):
    return nn.GroupNorm(32 if c % 32 == 0 else (8 if c % 8 == 0 else 1), c)


def _zero(m):
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


def emb_tempo(t, dim):
    """t (B,) em [0, 1] -> (B, dim), codificação senoidal."""
    meia = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(meia, device=t.device, dtype=torch.float32) / meia)
    a = (t.float()[:, None] * 1000.0) * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class BlocoRes(nn.Module):
    def __init__(self, ci, co, de, drop=0.0):
        super().__init__()
        self.n1, self.c1 = _gn(ci), nn.Conv2d(ci, co, 3, padding=1)
        self.e = nn.Linear(de, 2 * co)
        self.n2, self.drop = _gn(co), nn.Dropout(drop)
        self.c2 = _zero(nn.Conv2d(co, co, 3, padding=1))
        self.atalho = nn.Identity() if ci == co else nn.Conv2d(ci, co, 1)

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        esc, desl = self.e(F.silu(emb))[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + esc) + desl
        return self.atalho(x) + self.c2(self.drop(F.silu(h)))


class Atencao(nn.Module):
    def __init__(self, c, cabecas=4):
        super().__init__()
        self.n, self.cab = _gn(c), cabecas
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.saida = _zero(nn.Conv2d(c, c, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.cab, C // self.cab, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)
        return x + self.saida(h.transpose(-1, -2).reshape(B, C, H, W))


class Nivel(nn.Module):
    def __init__(self, ci, co, de, atencao, drop):
        super().__init__()
        self.res = BlocoRes(ci, co, de, drop)
        self.at = Atencao(co) if atencao else nn.Identity()

    def forward(self, x, emb):
        return self.at(self.res(x, emb))


class Desce(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.c(x)


class Sobe(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNetCond(nn.Module):
    """U-Net condicionada. tamanho deve ser divisível por 2^(len(mult) - 1)."""

    def __init__(self, canais, d_cond, n_classes=0, tamanho=64, largura=64, mult=(1, 2, 3, 4), blocos=2,
                 res_atencao=(16, 8), drop=0.0):
        super().__init__()
        self.canais, self.d_cond, self.n_classes, self.tamanho = canais, d_cond, n_classes, tamanho
        de = 4 * largura
        self.largura, self.mult, self.blocos = largura, tuple(mult), blocos
        self.t_mlp = nn.Sequential(nn.Linear(largura, de), nn.SiLU(), nn.Linear(de, de))
        self.c_mlp = nn.Sequential(nn.Linear(d_cond, de), nn.SiLU(), nn.Linear(de, de))
        self.nulo = nn.Parameter(torch.zeros(d_cond))                             # condição vazia (guia)
        self.classe = nn.Embedding(n_classes + 1, de) if n_classes > 0 else None  # índice n_classes = sem classe
        self.entrada = nn.Conv2d(canais, largura, 3, padding=1)
        chs = [largura * m for m in mult]
        self.desce, pilha, ci, res = nn.ModuleList(), [largura], largura, tamanho
        for i, c in enumerate(chs):
            for _ in range(blocos):
                self.desce.append(Nivel(ci, c, de, res in res_atencao, drop))
                ci = c
                pilha.append(ci)
            if i < len(chs) - 1:
                self.desce.append(Desce(ci))
                pilha.append(ci)
                res //= 2
        self.meio1, self.meio_at, self.meio2 = BlocoRes(ci, ci, de, drop), Atencao(ci), BlocoRes(ci, ci, de, drop)
        self.sobe = nn.ModuleList()
        for i, c in reversed(list(enumerate(chs))):
            for _ in range(blocos + 1):
                self.sobe.append(Nivel(ci + pilha.pop(), c, de, res in res_atencao, drop))
                ci = c
            if i > 0:
                self.sobe.append(Sobe(ci))
                res *= 2
        self.saida = nn.Sequential(_gn(ci), nn.SiLU(), _zero(nn.Conv2d(ci, canais, 3, padding=1)))

    def forward(self, x, t, cond, y=None):
        emb = self.t_mlp(emb_tempo(t, self.largura)) + self.c_mlp(cond)
        if self.classe is not None:
            emb = emb + self.classe(y if y is not None else torch.full((len(x),), self.n_classes, device=x.device))
        h = self.entrada(x)
        hs = [h]
        for m in self.desce:
            h = m(h, emb) if isinstance(m, Nivel) else m(h)
            hs.append(h)
        h = self.meio2(self.meio_at(self.meio1(h, emb)), emb)
        for m in self.sobe:
            h = m(torch.cat([h, hs.pop()], 1), emb) if isinstance(m, Nivel) else m(h)
        return self.saida(h)


# ---------------------------------------------------------------- difusão (tempo contínuo, cosseno, predição de v)


def alfa_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


def _nula(model, cond, y, mascara):
    """Troca a condição (e a classe) pela vazia onde mascara é True."""
    cond = torch.where(mascara[:, None], model.nulo.to(cond.dtype).expand_as(cond), cond)
    if model.classe is not None and y is not None:
        y = torch.where(mascara, torch.full_like(y, model.n_classes), y)
    return cond, y


def perda_v(model, x0, cond, y, p_sem_cond, t=None, eps=None):
    """Sorteia t e o ruído, suja a imagem e mede o erro da rede em prever v."""
    B = x0.size(0)
    if t is None:
        t = torch.rand(B, device=x0.device).clamp_(1e-4, 1 - 1e-4)
    if eps is None:
        eps = torch.randn_like(x0)
    a, s = (v[:, None, None, None] for v in alfa_sigma(t))
    xt, v = a * x0 + s * eps, a * eps - s * x0
    if p_sem_cond > 0:
        cond, y = _nula(model, cond, y, torch.rand(B, device=x0.device) < p_sem_cond)
    with autocast(x0.device):
        v_hat = model(xt, t, cond, y)
    return F.mse_loss(v_hat.float(), v)


@torch.no_grad()
def amostrar(model, cond, y=None, passos=50, guia=2.0, gerador=None):
    """cond (B, D) padronizado, no device do modelo -> imagens (B, C, H, W) em [-1, 1]. DDIM com guia."""
    dev, B = cond.device, cond.size(0)
    x = torch.randn(B, model.canais, model.tamanho, model.tamanho, generator=gerador).to(dev)
    ts = torch.linspace(1.0, 0.0, passos + 1, device=dev)
    com_guia = guia != 1.0
    if com_guia:
        cond_n, y_n = _nula(model, cond, y, torch.ones(B, dtype=torch.bool, device=dev))
    x0 = x
    for i in range(passos):
        t, s = ts[i].expand(B), ts[i + 1]
        a_t, s_t = (v[:, None, None, None] for v in alfa_sigma(t))
        with autocast(dev):
            if com_guia:
                yy = torch.cat([y, y_n]) if y is not None else None
                v_c, v_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([cond, cond_n]), yy).float().chunk(2)
                v = v_u + guia * (v_c - v_u)
            else:
                v = model(x, t, cond, y).float()
        x0 = (a_t * x - s_t * v).clamp(-1, 1)
        eps = (x - a_t * x0) / s_t
        a_s, s_s = alfa_sigma(s)
        x = a_s * x0 + s_s * eps
    return x0


@torch.no_grad()
def gerar(model, E_pad, y=None, passos=50, guia=2.0, device="cpu", lote=256, seed=0, log=None):
    """Embeddings padronizados (N, D) [+ rótulos, se o modelo usa a classe] -> imagens uint8 (N, H, W[, C]).
    A mesma chamada com a mesma semente dá as mesmas imagens."""
    model.eval()
    out, t0 = [], time.time()
    for k, i in enumerate(range(0, len(E_pad), lote)):
        g = torch.Generator().manual_seed(seed * 100003 + k)
        c = torch.as_tensor(E_pad[i:i + lote], device=device, dtype=torch.float32)
        yy = torch.as_tensor(y[i:i + lote], device=device) if (y is not None and model.classe is not None) else None
        out.append(para_u8(amostrar(model, c, yy, passos, guia, g)))
        if log and (k % 10 == 0 or i + lote >= len(E_pad)):
            feito = min(i + lote, len(E_pad))
            log(f"  imagens: {feito}/{len(E_pad)} ({time.time() - t0:.0f} s; "
                f"~{(time.time() - t0) / feito * (len(E_pad) - feito) / 60:.0f} min restantes)")
    return np.concatenate(out)


class EMA:
    """Média móvel exponencial dos pesos (é o modelo usado para gerar)."""

    def __init__(self, model, decaimento=0.999):
        self.modelo = copy.deepcopy(model).eval().requires_grad_(False)
        self.d, self.n = decaimento, 0

    @torch.no_grad()
    def atualiza(self, model):
        self.n += 1
        d = min(self.d, (1 + self.n) / (10 + self.n))
        for pe, p in zip(self.modelo.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.modelo.buffers(), model.buffers()):
            be.copy_(b)


@torch.no_grad()
def perda_validacao(model, X_u8, E_pad, y, device, seed=0, lote=128):
    """Perda de v com t e ruído fixos (para as épocas serem comparáveis). Só acompanhamento."""
    g = torch.Generator().manual_seed(seed)
    t_all = torch.rand(len(X_u8), generator=g).clamp_(1e-4, 1 - 1e-4)
    tot = 0.0
    for i in range(0, len(X_u8), lote):
        x0 = para_menos1_1(X_u8[i:i + lote], device)
        eps = torch.randn(x0.shape, generator=g).to(device)
        c = torch.as_tensor(E_pad[i:i + lote], device=device)
        yy = torch.as_tensor(y[i:i + lote], device=device) if model.classe is not None else None
        tot += perda_v(model, x0, c, yy, 0.0, t_all[i:i + lote].to(device), eps).item() * len(x0)
    return tot / len(X_u8)


def treinar(X_u8, E_pad, y, Xv_u8, Ev_pad, yv, canais, n_classes_cond=0, tamanho=64, largura=64, mult=(1, 2, 3, 4),
            blocos=2, epocas=20, lote=128, lr=2e-4, ema=0.999, p_sem_cond=0.1, aquecimento=1000, seed=0,
            device="cpu", log=print, ao_fim_da_epoca=None):
    """Treina nas imagens dadas, condicionando nos embeddings padronizados. Devolve (modelo EMA, histórico, nº de
    parâmetros). ao_fim_da_epoca(modelo_ema, época) é chamada depois de cada época (o treinar.py salva ali)."""
    torch.manual_seed(seed)
    model = UNetCond(canais, E_pad.shape[1], n_classes_cond, tamanho, largura, mult, blocos).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"  difusão: U-Net com {n_par / 1e6:.1f} M parâmetros; condição de {E_pad.shape[1]} números"
        + (" + classe" if n_classes_cond else "") + f"; imagens {tamanho}x{tamanho}x{canais}; {len(X_u8)} de treino")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    media = EMA(model, ema)
    g = torch.Generator().manual_seed(seed)
    hist, passo = [], 0
    for ep in range(1, epocas + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(X_u8), generator=g).numpy()
        soma, nb = 0.0, 0
        for i in range(0, max(len(perm) - lote + 1, 1), lote):  # descarta o último lote incompleto
            idx = np.sort(perm[i:i + lote])
            x0 = para_menos1_1(X_u8[idx], device)
            c = torch.as_tensor(E_pad[idx], device=device)
            yy = torch.as_tensor(y[idx], device=device) if n_classes_cond else None
            for gr in opt.param_groups:  # aquecimento linear da taxa de aprendizado
                gr["lr"] = lr * min(1.0, (passo + 1) / aquecimento)
            loss = perda_v(model, x0, c, yy, p_sem_cond)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            media.atualiza(model)
            soma += loss.item()
            nb += 1
            passo += 1
        val = perda_validacao(media.modelo, Xv_u8, Ev_pad, yv, device, seed=seed)
        hist.append({"epoca": ep, "treino_perda_v": soma / max(nb, 1), "val_perda_v_ema": val, "passos": passo,
                     "segundos": time.time() - t0})
        log(f"  difusão época {ep:3d}/{epocas}  treino {soma / max(nb, 1):.4f}  val (EMA) {val:.4f}  "
            f"{time.time() - t0:6.1f} s")
        if ao_fim_da_epoca is not None:
            ao_fim_da_epoca(media.modelo, ep)
    return media.modelo, hist, n_par


def salvar(model, caminho, media, desvio, **extra):
    """media e desvio: a padronização dos embeddings que a difusão espera (vão junto com os pesos)."""
    torch.save({"state_dict": model.state_dict(), "canais": model.canais, "d_cond": model.d_cond,
                "n_classes": model.n_classes, "tamanho": model.tamanho, "largura": model.largura, "mult": model.mult,
                "blocos": model.blocos,
                "media": np.asarray(media, np.float32), "desvio": np.asarray(desvio, np.float32), **extra}, caminho)


def carregar(caminho, device):
    """Também lê o difusao.pt do gerador de cvae_embeddings/ (mesmo formato)."""
    ck = torch.load(caminho, map_location=device, weights_only=False)
    model = UNetCond(ck["canais"], ck["d_cond"], ck["n_classes"], ck["tamanho"], ck["largura"], tuple(ck["mult"]),
                     ck["blocos"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
