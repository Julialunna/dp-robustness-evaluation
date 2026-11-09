"""extrator.py: embeddings DINOv2 congelados (o token CLS), com o pré-processamento do dp_FL.

Pré-processamento (o de dp_FL/extractor.py): pixels em [0, 1], imagem de 1 canal repetida em 3, redimensionamento
bilinear para 224, normalização ImageNet, token CLS na saída.

Backbones:
  dinov2_vits14  384 números, o do dp_FL; vem do torch.hub (precisa de internet na primeira vez; cache em TORCH_HOME).
  dinov2_vitb14  768 números, sem internet: lê os pesos de <data>/metadata/weights/dinov2_vitb14-*.pth (o dataset do
                 Flautim tem esse arquivo) e monta o modelo com o timm.
"""
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def construir(nome, data_dir, device):
    if nome == "dinov2_vits14":
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, skip_validation=True)
        except TypeError:  # versões antigas do torch.hub não aceitam skip_validation
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
    elif nome == "dinov2_vitb14":
        hits = sorted(glob.glob(os.path.join(data_dir, "metadata", "weights", "dinov2_vitb14-*.pth")))
        if not hits:
            raise SystemExit(f"dinov2_vitb14: sem pesos em {data_dir}/metadata/weights (use dinov2_vits14, com internet)")
        import timm
        from timm.models.vision_transformer import checkpoint_filter_fn
        sd = torch.load(hits[-1], map_location="cpu", weights_only=True)
        model = timm.create_model("vit_base_patch14_dinov2", pretrained=False, num_classes=0, img_size=224)
        model.load_state_dict(checkpoint_filter_fn(sd, model), strict=True)
    else:
        raise SystemExit(f"backbone desconhecido: {nome} (opções: {', '.join(DIMS)})")
    return model.eval().requires_grad_(False).to(device)


@torch.no_grad()
def _um_lote(model, imagens, device):
    """(n, H, W[, C]) uint8 -> (n, D) float32."""
    x = torch.from_numpy(np.ascontiguousarray(imagens)).to(device).float().div_(255.0)
    x = x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    return model(x).float().cpu()


def extrair(model, imagens, device, lote=256, rotulo=""):
    """(N, H, W[, C]) uint8 -> (N, D) float32, em lotes, com progresso no log."""
    out, t0 = [], time.time()
    for k, i in enumerate(range(0, len(imagens), lote)):
        out.append(_um_lote(model, imagens[i:i + lote], device))
        if rotulo and k % 50 == 0:
            print(f"  embeddings {rotulo}: {min(i + lote, len(imagens))}/{len(imagens)} ({time.time() - t0:.0f} s)",
                  flush=True)
    return torch.cat(out).numpy().astype(np.float32)
