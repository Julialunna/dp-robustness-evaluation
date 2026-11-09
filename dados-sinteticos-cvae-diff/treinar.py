"""treinar.py: treina o gerador de imagens sintéticas em dois estágios (sem DP).

  estágio 1  CVAE nos embeddings DINOv2: aprende p(embedding | classe)            (cvae.py)
  estágio 2  difusão condicionada no embedding: aprende p(imagem | embedding)     (difusao.py)
  sintético  classe y -> embedding e = CVAE(z, y) -> imagem = difusão(e)          (gerar.py)

Os dois estágios treinam no split de treino inteiro. A validação só acompanha: o CVAE guarda a época de menor perda na
validação; a difusão usa os pesos médios (EMA) do fim do treino. Os embeddings ficam em cache em <saida>/emb/ e são
reaproveitados se você rodar de novo com o mesmo dataset, tamanho e backbone.

Exemplos:
  python treinar.py --flag pathmnist                       # padrão: 64x64, 30 épocas de CVAE, 20 de difusão
  python treinar.py --flag bloodmnist --limite 300 --epocas 1 --epocas-cvae 2 --largura 32 --mult 1,1,1,1 --blocos 1 --passos 4
                                                           # teste rápido (roda em CPU)
Saída em <saida>/modelos/<flag>/:
  cvae.pt, difusao.pt                    os dois modelos (o gerar.py lê esta pasta)
  historico_cvae.csv, historico_difusao.csv
  amostras.png                           algumas imagens geradas no fim, por classe, para olhar
  config.json                            argumentos, tempos e versões
A difusão é salva no fim de cada época: se o job cair no meio, a última época completa está em difusao.pt.
"""
import argparse
import csv
import json
import os
import platform
import sys
import time

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import cvae as cv  # noqa: E402
import dados  # noqa: E402
import difusao as df  # noqa: E402
import extrator  # noqa: E402
import gerar as gr  # noqa: E402


def args_():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flag", default="pathmnist", choices=dados.FLAGS, help="qual dataset do MedMNIST")
    p.add_argument("--size", type=int, default=64, choices=[64, 128], help="tamanho das imagens (o do .npz)")
    p.add_argument("--data", default=os.environ.get("SINT_DATA", os.path.join(AQUI, "dataset")),
                   help="pasta com <flag>_<size>.npz (e metadata/)")
    p.add_argument("--saida", default=os.environ.get("SINT_OUTPUT", os.path.join(os.path.dirname(AQUI), "output")))
    p.add_argument("--backbone", default=os.environ.get("BACKBONE", "dinov2_vits14"), choices=list(extrator.DIMS))
    p.add_argument("--limite", type=int, default=0, help="usa só esse número de linhas por split (testes rápidos)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--auto-download", type=int, default=int(os.environ.get("AUTO_DOWNLOAD", "1")))
    g1 = p.add_argument_group("estágio 1: CVAE nos embeddings")
    g1.add_argument("--epocas-cvae", type=int, default=30)
    g1.add_argument("--latente", type=int, default=32)
    g1.add_argument("--oculta", type=int, default=256)
    g1.add_argument("--beta", type=float, default=1.0, help="peso do termo KL")
    g2 = p.add_argument_group("estágio 2: difusão condicionada no embedding")
    g2.add_argument("--epocas", type=int, default=20,
                    help="20 épocas nos 90 mil do PathMNIST = ~14 mil passos (o mesmo do teste que funcionou)")
    g2.add_argument("--largura", type=int, default=64, help="canais da primeira camada da U-Net")
    g2.add_argument("--mult", default="1,2,3,4", help="multiplicadores de canal por resolução (64, 32, 16, 8)")
    g2.add_argument("--blocos", type=int, default=2, help="blocos residuais por resolução")
    g2.add_argument("--lote", type=int, default=128)
    g2.add_argument("--lr", type=float, default=2e-4)
    g2.add_argument("--ema", type=float, default=0.999)
    g2.add_argument("--p-sem-cond", type=float, default=0.1, help="fração do treino sem condição (para a guia)")
    g2.add_argument("--cond-classe", type=int, default=0, help="1: condiciona também na classe (padrão: só embedding)")
    g3 = p.add_argument_group("amostras de conferência no fim")
    g3.add_argument("--passos", type=int, default=50)
    g3.add_argument("--guia", type=float, default=2.0)
    g3.add_argument("--n-por-classe", type=int, default=8)
    return p.parse_args()


def log(*a):
    print(*a, flush=True)


def embeddings(a, X, device):
    """Embeddings de treino e validação, com cache em <saida>/emb/."""
    os.makedirs(os.path.join(a.saida, "emb"), exist_ok=True)
    suf = f"_lim{a.limite}_s{a.seed}" if a.limite else ""
    cache = os.path.join(a.saida, "emb", f"{a.flag}_{a.size}_{a.backbone}{suf}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        if all(np.array_equal(z[f"y_{s}"], X[s][1]) for s in ("train", "val")):
            log(f"  embeddings do cache: {cache}")
            return {s: z[f"X_{s}"] for s in ("train", "val")}
        log(f"  (cache {cache} não corresponde aos dados; extraindo de novo)")
    t0 = time.time()
    model = extrator.construir(a.backbone, a.data, device)
    E = {s: extrator.extrair(model, X[s][0], device, rotulo=f"{a.flag}/{s}") for s in ("train", "val")}
    del model
    np.savez(cache, **{f"X_{s}": E[s] for s in E}, **{f"y_{s}": X[s][1] for s in E})
    log(f"  embeddings extraídos em {time.time() - t0:.0f} s; cache em {cache}")
    return E


def salvar_csv(caminho, linhas):
    with open(caminho, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)


def main():
    a = args_()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        log(f"  GPU {torch.cuda.get_device_name(0)}")
    t_ini = time.time()
    mult = tuple(int(m) for m in a.mult.split(","))
    if a.size % 2 ** (len(mult) - 1):
        raise SystemExit(f"--size {a.size} não é divisível por 2^{len(mult) - 1} (exigido por --mult {a.mult})")
    nomes = dados.nomes_das_classes(a.flag, a.data)
    C = len(nomes)
    pasta = os.path.join(a.saida, "modelos", a.flag)
    os.makedirs(pasta, exist_ok=True)
    log(f"############ treino do gerador em dois estágios: {a.flag} {a.size}x{a.size}, {a.backbone}, "
        f"device {device} ############")

    # ---- dados
    npz = dados.caminho_npz(a.flag, a.size, a.data, os.path.join(a.saida, "dados"), bool(a.auto_download))
    X = {s: dados.carregar(npz, s, a.limite, a.seed) for s in ("train", "val")}
    (Xtr, ytr), (Xva, yva) = X["train"], X["val"]
    canais = 1 if Xtr.ndim == 3 else Xtr.shape[-1]
    log(f"  {npz}: treino {Xtr.shape}, validação {Xva.shape}, {C} classes")
    E = embeddings(a, X, device)
    Etr, Eva = E["train"], E["val"]

    # ---- estágio 1: CVAE nos embeddings
    t0 = time.time()
    cvae, hist_c, ep_c = cv.treinar(Etr, ytr, Eva, yva, C, a.latente, a.oculta, a.epocas_cvae, beta=a.beta,
                                    seed=a.seed, device=device, log=log)
    seg_cvae = time.time() - t0
    cv.salvar(cvae, os.path.join(pasta, "cvae.pt"), beta=a.beta, backbone=a.backbone, flag=a.flag, classes=nomes,
              contagem_classes=np.bincount(ytr, minlength=C).tolist(), epoca=ep_c)
    salvar_csv(os.path.join(pasta, "historico_cvae.csv"), hist_c)
    log(f"  CVAE: melhor época {ep_c} de {a.epocas_cvae} ({seg_cvae:.0f} s) -> {pasta}/cvae.pt")

    # ---- estágio 2: difusão condicionada no embedding (padronizado com média e desvio do treino)
    media, desvio = Etr.mean(0), Etr.std(0) + 1e-6

    def pad(Z):
        return ((Z - media) / desvio).astype(np.float32)

    sub_v = np.sort(np.random.default_rng(a.seed).choice(len(Xva), min(512, len(Xva)), replace=False))
    extra = dict(backbone=a.backbone, flag=a.flag, classes=nomes, passos=a.passos, guia=a.guia)

    def salva_epoca(modelo_ema, ep):
        df.salvar(modelo_ema, os.path.join(pasta, "difusao.pt"), media, desvio, epoca=ep, **extra)

    t0 = time.time()
    dif, hist_d, n_par = df.treinar(Xtr, pad(Etr), ytr, Xva[sub_v], pad(Eva[sub_v]), yva[sub_v], canais,
                                    C if a.cond_classe else 0, a.size, a.largura, mult, a.blocos, a.epocas, a.lote,
                                    a.lr, a.ema, a.p_sem_cond, seed=a.seed, device=device, log=log,
                                    ao_fim_da_epoca=salva_epoca)
    seg_dif = time.time() - t0
    salvar_csv(os.path.join(pasta, "historico_difusao.csv"), hist_d)
    log(f"  difusão: {a.epocas} épocas em {seg_dif / 60:.0f} min -> {pasta}/difusao.pt")

    # ---- algumas amostras para conferir
    t0 = time.time()
    _, _, _, ck_d = gr.carregar_modelos(os.path.join(pasta, "cvae.pt"), os.path.join(pasta, "difusao.pt"), device)
    y = np.repeat(np.arange(C), a.n_por_classe)
    imgs, _ = gr.sinteticos(cvae, dif, ck_d, y, a.passos, a.guia, device, seed=a.seed)
    gr.figura_amostras(os.path.join(pasta, "amostras.png"), imgs, y, nomes,
                       f"{a.flag}: amostras do gerador no fim do treino", k=a.n_por_classe)
    log(f"  {len(y)} amostras de conferência em {time.time() - t0:.0f} s -> {pasta}/amostras.png")

    with open(os.path.join(pasta, "config.json"), "w") as fh:
        json.dump({"args": vars(a), "npz": npz, "device": device, "n_treino": int(len(ytr)),
                   "cvae_melhor_epoca": ep_c, "difusao_parametros": n_par,
                   "segundos": {"cvae": seg_cvae, "difusao": seg_dif, "total": time.time() - t_ini},
                   "final": {"cvae_val_perda": hist_c[ep_c - 1]["val_perda"],
                             "difusao_val_perda_v": hist_d[-1]["val_perda_v_ema"]},
                   "versoes": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__}}, fh, indent=2, ensure_ascii=False)
    log(f"-> modelos em {pasta} ({(time.time() - t_ini) / 60:.0f} min no total). Para gerar: "
        f"python gerar.py --modelos {pasta}")


if __name__ == "__main__":
    main()
