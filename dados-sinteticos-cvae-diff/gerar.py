"""gerar.py: gera imagens sintéticas (e os embeddings correspondentes) com os modelos salvos pelo treinar.py.

  classe y  ->  embedding e = CVAE(z, y), z ~ N(0, I)  ->  imagem = difusão(e)

Exemplos:
  python gerar.py --modelos ../output/modelos/pathmnist                    # 10 000 sintéticos, classes como no treino
  python gerar.py --modelos ../output/modelos/pathmnist --n 900 --balanceado
  python gerar.py --cvae caminho/cvae_privado.pt --difusao caminho/difusao.pt --flag pathmnist
                                                                         # modelos do gerador de cvae_embeddings/
Saída em <saida>/sinteticos/:
  sinteticos_<flag>.npz   images (N, H, W[, C]) uint8, labels (N,) int64, embeddings (N, D) float32, classes (nomes)
  amostras_<flag>.png     algumas imagens por classe
Tempo: ~6 imagens por segundo na GPU MIG 1g.18gb do Flautim (50 passos, guia 2), ou seja, ~27 min para 10 000.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import cvae as cv  # noqa: E402
import dados  # noqa: E402
import difusao as df  # noqa: E402


def log(*a):
    print(*a, flush=True)


def carregar_modelos(cvae_pt, difusao_pt, device):
    cvae, ck_c = cv.carregar(cvae_pt, device)
    dif, ck_d = df.carregar(difusao_pt, device)
    if ck_c["d"] != ck_d["d_cond"]:
        raise SystemExit(f"o CVAE gera {ck_c['d']} números e a difusão espera {ck_d['d_cond']}: modelos de "
                         "backbones diferentes")
    return cvae, ck_c, dif, ck_d


def sinteticos(cvae, dif, ck_d, y, passos=50, guia=2.0, device="cpu", lote=256, seed=0, log=None):
    """Rótulos y (N,) -> (imagens uint8, embeddings float32 na escala original do DINOv2)."""
    E = cvae.gerar(torch.as_tensor(y), gerador=torch.Generator().manual_seed(seed)).numpy().astype(np.float32)
    E_pad = ((E - ck_d["media"]) / ck_d["desvio"]).astype(np.float32)
    imgs = df.gerar(dif, E_pad, y, passos=passos, guia=guia, device=device, lote=lote, seed=seed, log=log)
    return imgs, E


def rotulos(n, n_classes, contagem=None, seed=0):
    """n rótulos embaralhados; com contagem (exemplos por classe no treino), na mesma proporção; senão, balanceados."""
    p = np.ones(n_classes) if contagem is None else np.asarray(contagem, float)
    p = p / p.sum()
    k = np.floor(p * n).astype(int)
    k[np.argsort(-(p * n - k))[:n - k.sum()]] += 1  # maiores restos: a soma dá exatamente n
    y = np.repeat(np.arange(n_classes), k)
    return np.random.default_rng(seed).permutation(y).astype(np.int64)


def figura_amostras(caminho, imgs, y, nomes, titulo, k=8):
    """Linhas = classes, k imagens por linha."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = len(nomes)
    k = max(1, min(k, max(int((y == c).sum()) for c in range(C))))
    W, H, esq = max(k * 0.9 + 2.6, 6.5), C * 0.9 + 0.5, 2.6
    fig, axs = plt.subplots(C, k, figsize=(W, H), squeeze=False,
                            gridspec_kw={"wspace": 0.04, "hspace": 0.06, "left": esq / W, "right": 0.995,
                                         "top": 1 - 0.45 / H, "bottom": 0.05 / H})
    for c in range(C):
        idx = np.flatnonzero(y == c)[:k]
        for j in range(k):
            ax = axs[c, j]
            ax.axis("off")
            if j < len(idx):
                im = imgs[idx[j]]
                ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
        nome = nomes[c] if len(nomes[c]) <= 30 else nomes[c][:29] + "…"
        axs[c, 0].text(-0.1, 0.5, f"{c}  {nome}", transform=axs[c, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=10)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def args_():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelos", default="", help="pasta com cvae.pt e difusao.pt (saída do treinar.py)")
    p.add_argument("--cvae", default="", help="caminho do CVAE (em vez de --modelos)")
    p.add_argument("--difusao", default="", help="caminho da difusão (em vez de --modelos)")
    p.add_argument("--flag", default="", help="nome do dataset (só para checkpoints que não o guardam)")
    p.add_argument("--data", default=os.environ.get("SINT_DATA", os.path.join(AQUI, "dataset")),
                   help="só para ler os nomes das classes de <data>/metadata, se existir")
    p.add_argument("--saida", default=os.environ.get("SINT_OUTPUT", os.path.join(os.path.dirname(AQUI), "output")))
    p.add_argument("--n", type=int, default=int(os.environ.get("N_SINT", "10000")), help="número de sintéticos")
    p.add_argument("--balanceado", action="store_true", help="o mesmo número por classe (padrão: proporção do treino)")
    p.add_argument("--passos", type=int, default=50, help="passos DDIM (menos = mais rápido, imagens piores)")
    p.add_argument("--guia", type=float, default=2.0, help="escala da guia sem classificador (1 = sem guia)")
    p.add_argument("--lote", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    a = args_()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    cvae_pt = a.cvae or os.path.join(a.modelos, "cvae.pt")
    dif_pt = a.difusao or os.path.join(a.modelos, "difusao.pt")
    for p in (cvae_pt, dif_pt):
        if not os.path.exists(p):
            raise SystemExit(f"não achei {p} (passe --modelos <pasta> ou --cvae e --difusao)")
    t0 = time.time()
    cvae, ck_c, dif, ck_d = carregar_modelos(cvae_pt, dif_pt, device)
    flag = ck_c.get("flag") or a.flag or "dados"
    nomes = ck_c.get("classes") or (dados.nomes_das_classes(flag, a.data) if flag in dados.FLAGS
                                    else [str(c) for c in range(ck_c["n_classes"])])
    contagem = None if a.balanceado else ck_c.get("contagem_classes")
    y = rotulos(a.n, ck_c["n_classes"], contagem, a.seed)
    log(f"############ gerando {a.n} sintéticos: {flag}, {ck_d['tamanho']}x{ck_d['tamanho']}x{ck_d['canais']}, "
        f"device {device} ############")
    log(f"  modelos: {cvae_pt} e {dif_pt}")
    log("  classes: " + ("balanceadas" if contagem is None else "na proporção do treino") + " -> "
        + ", ".join(f"{c}: {(y == c).sum()}" for c in range(ck_c["n_classes"])))
    imgs, E = sinteticos(cvae, dif, ck_d, y, a.passos, a.guia, device, a.lote, a.seed, log)
    pasta = os.path.join(a.saida, "sinteticos")
    os.makedirs(pasta, exist_ok=True)
    arq = os.path.join(pasta, f"sinteticos_{flag}.npz")
    np.savez_compressed(arq, images=imgs, labels=y, embeddings=E, classes=np.array(nomes))
    figura_amostras(os.path.join(pasta, f"amostras_{flag}.png"), imgs, y, nomes,
                    f"{flag}: imagens sintéticas (CVAE de embeddings + difusão), {a.passos} passos, guia {a.guia}")
    log(f"-> {arq}: images {imgs.shape}, labels {y.shape}, embeddings {E.shape} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
