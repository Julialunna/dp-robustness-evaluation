"""
Módulo de Extração de Features.

Define a interface padrão `BaseFeatureExtractor`, a implementação oficial `DinoV2Extractor`
e funções auxiliares para extração em lote e cache de embeddings.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config


class BaseFeatureExtractor(nn.Module, ABC):
    """Interface abstrata base para qualquer extrator de características."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensão do vetor de embedding gerado pelo extrator."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Tamanho da imagem quadrada de entrada (ex: 224 para 224x224)."""
        pass

    @abstractmethod
    def resize_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Ajusta canais e redimensiona as imagens mantendo valores em [0, 1]."""
        pass

    @abstractmethod
    def normalize_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Aplica a normalização estatística necessária para o extrator (ex: ImageNet)."""
        pass

    def preprocess_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Aplica redimensionamento e normalização necessários para o modelo."""
        images = self.resize_images(images, device=device)
        return self.normalize_images(images, device=device)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Gera os embeddings para o batch de imagens pré-processadas."""
        pass


class DinoV2Extractor(BaseFeatureExtractor):
    """Extrator de features baseado no DINOv2 (Vision Transformer auto-supervisionado)."""

    MODEL_HUB_NAMES = {
        "dinov2_s": "dinov2_vits14",
        "dinov2_vits14": "dinov2_vits14",
        "dinov2_vitb14": "dinov2_vitb14",
        "dinov2_vitl14": "dinov2_vitl14",
        "dinov2_vitg14": "dinov2_vitg14",
    }

    EMBEDDING_DIMS = {
        "dinov2_s": 384,
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str = "dinov2_s", image_size: int = 224) -> None:
        super().__init__()
        hub_name = self.MODEL_HUB_NAMES.get(model_name, model_name)
        self._embedding_dim = self.EMBEDDING_DIMS.get(model_name, 384)
        self._input_size = image_size

        # Carrega o modelo pré-treinado oficial do DINOv2 via PyTorch Hub com fallback offline
        try:
            self.model = torch.hub.load(
                "facebookresearch/dinov2", hub_name, trust_repo=True, skip_validation=True
            )
        except Exception:
            self.model = torch.hub.load("facebookresearch/dinov2", hub_name, trust_repo=True)
        self.model.eval()

        # Congela todos os parâmetros (extrator de features estático)
        for param in self.model.parameters():
            param.requires_grad = False

        # Constantes de normalização ImageNet
        self.register_buffer(
            "imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def input_size(self) -> int:
        return self._input_size

    def resize_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        images = images.to(device).float()

        # Adiciona dimensão de canal se necessário (B, H, W) -> (B, 1, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(1)

        # Converte imagens de 1 canal (ex: MNIST) para 3 canais RGB
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)

        # Redimensiona para o tamanho de entrada do DINO (224x224)
        if images.shape[-2:] != (self._input_size, self._input_size):
            images = F.interpolate(
                images,
                size=(self._input_size, self._input_size),
                mode="bilinear",
                align_corners=False,
            )

        return images

    def normalize_images(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        images = images.to(device).float()
        mean = self.imagenet_mean.to(device=device, dtype=images.dtype)
        std = self.imagenet_std.to(device=device, dtype=images.dtype)
        return (images - mean) / std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


_EXTRACTOR_CACHE: Dict[Tuple[str, str], BaseFeatureExtractor] = {}


def get_feature_extractor(
    name: str = config.EXTRACTOR_NAME,
    device: Optional[torch.device] = None,
) -> BaseFeatureExtractor:
    """
    Factory para instanciação de extratores de features.
    
    Utiliza cache singleton em memória para evitar recarregar o ViT repetidamente
    entre instâncias de clientes federados.
    """
    dev = device or torch.device("cpu")
    key = (name, str(dev))
    if key not in _EXTRACTOR_CACHE:
        if name.startswith("dinov2"):
            ext = DinoV2Extractor(model_name=name, image_size=config.IMAGE_SIZE)
        else:
            raise ValueError(
                f"Extrator '{name}' não reconhecido. Opções válidas: 'dinov2_s', 'dinov2_vitb14', etc."
            )
        ext.to(dev)
        _EXTRACTOR_CACHE[key] = ext
    return _EXTRACTOR_CACHE[key]


@torch.no_grad()
def extract_embeddings_from_loader(
    extractor: BaseFeatureExtractor,
    loader: DataLoader,
    device: torch.device,
    image_noise_std: float = 0.0,
    image_noise_seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extrai todos os embeddings e rótulos de um DataLoader de imagens.
    Suporta injeção de ruído gaussiano (para avaliação de robustez).
    """
    extractor.to(device)
    extractor.eval()

    generator = None
    if image_noise_seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(image_noise_seed)

    embeddings_list = []
    labels_list = []

    for batch in loader:
        images = batch["image"].float()

        # 1. Ajusta canais (1->3) e redimensiona para o tamanho do extrator (224x224) em [0, 1]
        images = extractor.resize_images(images, device=device)

        # 2. Injeção de ruído gaussiano nas imagens redimensionadas em [0, 1]
        if image_noise_std > 0:
            noise = torch.randn(
                images.shape,
                generator=generator,
                device="cpu",
                dtype=images.dtype,
            ).to(device)
            images = torch.clamp(images + image_noise_std * noise, 0.0, 1.0)

        # 3. Normalização estatística (ImageNet mean/std) e extração de features
        images = extractor.normalize_images(images, device=device)
        embeddings = extractor(images)

        # Prepara rótulos
        raw_labels = batch["label"]
        if isinstance(raw_labels, list):
            labels = torch.tensor([int(l[0]) if isinstance(l, list) else int(l) for l in raw_labels], dtype=torch.long)
        elif isinstance(raw_labels, torch.Tensor):
            labels = raw_labels.to(dtype=torch.long).reshape(-1)
        else:
            labels = torch.tensor(raw_labels, dtype=torch.long).reshape(-1)

        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())

    return torch.cat(embeddings_list, dim=0), torch.cat(labels_list, dim=0)


def build_or_load_embedding_loader(
    cache_path: Path | str,
    extractor: BaseFeatureExtractor,
    image_loader: DataLoader,
    device: torch.device,
    metadata: Dict,
    batch_size: int = config.BATCH_SIZE,
    image_noise_std: float = 0.0,
    image_noise_seed: Optional[int] = None,
) -> DataLoader:
    """Carrega embeddings do cache em disco se os metadados forem compatíveis; caso contrário, extrai e salva."""
    cache_path = Path(cache_path)

    if cache_path.exists():
        try:
            cached_data = torch.load(cache_path, map_location="cpu")
            if cached_data.get("metadata", {}) == metadata:
                dataset = TensorDataset(
                    cached_data["embeddings"].float(),
                    cached_data["labels"].long(),
                )
                return DataLoader(dataset, batch_size=batch_size, shuffle=False)
        except Exception:
            pass  # Se o arquivo estiver corrompido, regenera

    # Extrai novos embeddings
    embeddings, labels = extract_embeddings_from_loader(
        extractor=extractor,
        loader=image_loader,
        device=device,
        image_noise_std=image_noise_std,
        image_noise_seed=image_noise_seed,
    )

    # Salva no cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings.cpu(),
            "labels": labels.cpu(),
            "metadata": metadata,
        },
        cache_path,
    )

    dataset = TensorDataset(embeddings.float(), labels.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
