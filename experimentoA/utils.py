"""
Funções Utilitárias Gerais: Gerenciamento de dispositivos, sementes e recursos.
"""

import gc
import random
from typing import Any

import numpy as np
import torch


def get_device() -> torch.device:
    """Retorna o dispositivo computacional ideal disponível (CUDA, MPS ou CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    """Configura a semente pseudo-aleatória em todas as bibliotecas para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_python_scalar(value: Any) -> Any:
    """Converte tensores ou tipos escalares do NumPy para tipos nativos do Python (int, float)."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def release_resources() -> None:
    """Libera memória GPU/MPS e encerra processos em segundo plano (como Ray)."""
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass