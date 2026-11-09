"""dados.py: lê o MedMNIST (.npz) e baixa do Zenodo o que faltar.

Arquivo esperado: <data>/<flag>_<size>.npz, no formato oficial do MedMNIST:
  train_images (N, H, W) ou (N, H, W, C) uint8 e train_labels (N, 1); o mesmo para val_* e test_*.
Se o arquivo não existir e auto_download=True, baixa a URL listada em metadata/<flag>.json, confere o md5 e salva em
<destino>. Os metadados (nomes das classes, canais, URLs) vêm de <data>/metadata/<flag>.json ou da cópia em ./metadata.
"""
import hashlib
import json
import os
import urllib.request

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
FLAGS = ("pathmnist", "bloodmnist", "dermamnist", "octmnist",
         "tissuemnist", "organamnist", "organcmnist", "organsmnist")


def metadados(flag, data_dir):
    for d in (os.path.join(data_dir, "metadata"), os.path.join(AQUI, "metadata")):
        p = os.path.join(d, f"{flag}.json")
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
    raise SystemExit(f"sem metadados para {flag!r} (procurei em {data_dir}/metadata e {AQUI}/metadata)")


def nomes_das_classes(flag, data_dir):
    meta = metadados(flag, data_dir)
    return [meta["labels"][str(c)] for c in range(int(meta["n_classes"]))]


def _md5(caminho, bloco=1 << 20):
    h = hashlib.md5()
    with open(caminho, "rb") as fh:
        for b in iter(lambda: fh.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def caminho_npz(flag, size, data_dir, destino, auto_download=True):
    """Devolve o caminho do <flag>_<size>.npz; procura em data_dir e em destino e, se faltar, baixa para destino."""
    nome = f"{flag}_{size}.npz"
    for d in (data_dir, destino):
        p = os.path.join(d, nome)
        if os.path.exists(p):
            return p
    if not auto_download:
        raise SystemExit(f"não achei {nome} em {data_dir} nem em {destino} (e o download automático está desligado)")
    spec = metadados(flag, data_dir).get("archives", {}).get(str(size))
    if spec is None:
        raise SystemExit(f"{flag}: os metadados não listam o arquivo {size}x{size}")
    os.makedirs(destino, exist_ok=True)
    p = os.path.join(destino, nome)
    print(f"[baixando] {spec['url']}", flush=True)
    urllib.request.urlretrieve(spec["url"], p + ".part")
    md5 = _md5(p + ".part")
    if spec.get("md5") and md5 != spec["md5"]:
        os.remove(p + ".part")
        raise SystemExit(f"{flag}: md5 {md5}, esperado {spec['md5']}; download descartado")
    os.replace(p + ".part", p)
    print(f"[ok] {p} (md5 conferido)", flush=True)
    return p


def carregar(caminho, split, limite=0, seed=0):
    """(imagens uint8 (N, H, W[, C]), rótulos int64 (N,)) de um split. limite > 0 sorteia esse número de linhas
    (sempre as mesmas para a mesma semente), para testes rápidos."""
    with np.load(caminho) as z:
        x = z[f"{split}_images"]
        y = np.asarray(z[f"{split}_labels"]).reshape(len(x), -1)[:, 0].astype(np.int64)
    if limite and len(x) > limite:
        idx = np.sort(np.random.default_rng(seed).choice(len(x), limite, replace=False))
        x, y = x[idx], y[idx]
    return x, y
