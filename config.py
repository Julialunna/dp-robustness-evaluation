"""
Configurações Globais do Projeto de Aprendizado Federado com DP-CVAE e DINOv2.

Este arquivo é a fonte única de verdade para todos os parâmetros do experimento.
Organizado por seções temáticas para facilitar alterações manuais.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ==============================================================================
# 1. DATASET E PARTICIONAMENTO
# ==============================================================================
# Opções de dataset: "pathmnist", "breastmnist", "mnist", etc.
DATASET = "pathmnist"

# Número de classes do dataset (ex: pathmnist=9, breastmnist=2, mnist=10)
NUM_CLASSES = 9

# Particionamento dos dados entre clientes federados: "dirichlet" (não-IID) ou "iid"
PARTITIONER = "dirichlet"

# Concentração Dirichlet (quanto menor, mais heterogênea a distribuição entre clientes)
DIRICHLET_ALPHA = 1.0


# ==============================================================================
# 2. EXTRATOR DE FEATURES (DINOv2)
# ==============================================================================
# Modelo extrator: "dinov2_s" (ou "dinov2_vits14"), "dinov2_vitb14", "dinov2_vitl14"
EXTRACTOR_NAME = "dinov2_s"

# Resolução de entrada esperada pelo extrator (DINOv2 trabalha com 224x224)
IMAGE_SIZE = 224

# Dimensões de embedding por arquitetura
EXTRACTOR_EMBEDDING_DIMS = {
    "dinov2_s": 384,
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# Dimensão do embedding correspondente ao extrator selecionado
EMBEDDING_DIM = EXTRACTOR_EMBEDDING_DIMS.get(EXTRACTOR_NAME, 384)


# ==============================================================================
# 3. CLASSIFICADOR DOWNSTREAM
# ==============================================================================
# Dimensão da camada oculta do EmbeddingClassifier
CLASSIFIER_HIDDEN_DIM = 128


# ==============================================================================
# 4. TREINAMENTO FEDERADO (FLOWER)
# ==============================================================================
# Número total de clientes particionados
NUM_PARTITIONS = 10

# Número de rodadas de comunicação federada
NUM_SERVER_ROUNDS = 20

# Fração de clientes selecionados para treino por rodada
FRACTION_FIT = 1.0

# Fração de clientes selecionados para avaliação por rodada
FRACTION_EVALUATE = 1.0

# Batch size para o classificador local e data loaders
BATCH_SIZE = 256

# Épocas locais de treino do classificador em cada rodada
LOCAL_EPOCHS = 1

# Taxa de aprendizado do classificador local
LR = 0.001


# ==============================================================================
# 5. MODALIDADE DE TREINO E DP-CVAE
# ==============================================================================
# Modalidade de treino do cliente:
# - "baseline": DINOv2 direto nos dados reais (sem CVAE)
# - "cvae_sem_dp": Treina CVAE local sem DP-SGD e gera dados sintéticos
# - "cvae_dp": Treina CVAE local com DP-SGD (Opacus) usando TARGET_EPSILON
TRAIN_MODE = "cvae_dp"

# Se True, habilita DP-SGD no CVAE (sincronizado automaticamente com TRAIN_MODE)
USE_LOCAL_DP_CVAE = (TRAIN_MODE == "cvae_dp")

# Hiperparâmetros de Privacidade Diferencial (Opacus)
TARGET_EPSILON = 5.0
TARGET_DELTA = 1e-5
MAX_GRAD_NORM = 1.2

# Hiperparâmetros da arquitetura do CVAE
CVAE_HIDDEN_DIM = 256
CVAE_LATENT_DIM = 32
CVAE_BATCH_SIZE = 256
CVAE_EPOCHS = 10
CVAE_LR = 0.001
CVAE_BETA = 0.1  # Peso da divergência KL (KLD) na loss do CVAE


# ==============================================================================
# 6. CACHE E REPRODUTIBILIDADE
# ==============================================================================
# Diretórios para persistir embeddings calculados e conjuntos sintéticos gerados
REAL_EMBEDDINGS_CACHE_DIR = "artifacts/real_embeddings"
SYNTHETIC_CACHE_DIR = "artifacts/synthetic_embeddings"
TEST_EMBEDDINGS_CACHE_DIR = "artifacts/test_embeddings"
SYNTHETIC_CACHE_VERSION = 4

# Controle de regeneração de cache
FORCE_REGENERATE_SYNTHETICS = False
RETRAIN_CVAE_EVERY_ROUND = False


# ==============================================================================
# 7. AVALIAÇÃO DE ROBUSTEZ A RUÍDO (SERVIDOR)
# ==============================================================================
# Semente para geração determinística do ruído de teste
EVAL_GAUSSIAN_NOISE_SEED = 1234

# Níveis de desvio padrão do ruído gaussiano adicionado às imagens no teste final:
# σ = 0.0 (limpo), σ1 = 0.10, σ2 = 0.25, σ3 = 0.50
EVAL_GAUSSIAN_NOISE_STD1 = 0.10
EVAL_GAUSSIAN_NOISE_STD2 = 0.25
EVAL_GAUSSIAN_NOISE_STD3 = 0.50


# ==============================================================================
# 8. SIMULAÇÃO E EXPERIMENTOS INDEPENDENTES
# ==============================================================================
# Número de repetições completas do experimento padrão
NUM_RUNS = 5

# Sementes padrão para o Grid Search (5 repetições)
GRID_SEEDS = [2026, 2027, 2028, 2029, 2030]
BASE_SEED = 2026


# ==============================================================================
# 9. SINCRONIZAÇÃO DINÂMICA ENTRE PROCESSOS (RAY / WORKERS / SPAWN)
# ==============================================================================
def get_env_vars() -> Dict[str, str]:
    """Retorna dicionário de variáveis de ambiente com a configuração atual."""
    return {
        "FLWR_DATASET": str(DATASET),
        "FLWR_PARTITIONER": str(PARTITIONER),
        "FLWR_DIRICHLET_ALPHA": str(DIRICHLET_ALPHA),
        "FLWR_TRAIN_MODE": str(TRAIN_MODE),
        "FLWR_TARGET_EPSILON": str(TARGET_EPSILON),
        "FLWR_NUM_SERVER_ROUNDS": str(NUM_SERVER_ROUNDS),
    }


def apply_config(
    dataset: Optional[str] = None,
    partitioner: Optional[str] = None,
    dirichlet_alpha: Optional[float] = None,
    train_mode: Optional[str] = None,
    target_epsilon: Optional[float] = None,
    num_server_rounds: Optional[int] = None,
) -> None:
    """
    Aplica configuração dinâmica válida para o processo principal E para todos
    os subprocessos/workers Ray através de variáveis de ambiente e arquivo ativo.
    """
    global DATASET, PARTITIONER, DIRICHLET_ALPHA, TRAIN_MODE, USE_LOCAL_DP_CVAE, TARGET_EPSILON, NUM_SERVER_ROUNDS, EMBEDDING_DIM

    if dataset is not None:
        DATASET = dataset
    if partitioner is not None:
        PARTITIONER = partitioner
    if dirichlet_alpha is not None:
        DIRICHLET_ALPHA = float(dirichlet_alpha)
    if train_mode is not None:
        TRAIN_MODE = train_mode
        USE_LOCAL_DP_CVAE = (TRAIN_MODE == "cvae_dp")
    if target_epsilon is not None:
        TARGET_EPSILON = float(target_epsilon)
    if num_server_rounds is not None:
        NUM_SERVER_ROUNDS = int(num_server_rounds)

    EMBEDDING_DIM = EXTRACTOR_EMBEDDING_DIMS.get(EXTRACTOR_NAME, 384)

    # 1. Atualiza variáveis de ambiente (herdadas por workers e Ray)
    for k, v in get_env_vars().items():
        os.environ[k] = v

    # 2. Persiste em arquivo de configuração ativa para subprocessos isolados
    active_cfg_path = Path("artifacts/.active_config.json")
    try:
        active_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        active_cfg_path.write_text(
            json.dumps(
                {
                    "DATASET": DATASET,
                    "PARTITIONER": PARTITIONER,
                    "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
                    "TRAIN_MODE": TRAIN_MODE,
                    "USE_LOCAL_DP_CVAE": USE_LOCAL_DP_CVAE,
                    "TARGET_EPSILON": TARGET_EPSILON,
                    "NUM_SERVER_ROUNDS": NUM_SERVER_ROUNDS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def clear_active_config() -> None:
    """Remove o arquivo de configuração ativa após o término da simulação."""
    active_cfg_path = Path("artifacts/.active_config.json")
    if active_cfg_path.exists():
        try:
            active_cfg_path.unlink()
        except Exception:
            pass
    for k in ["FLWR_DATASET", "FLWR_PARTITIONER", "FLWR_DIRICHLET_ALPHA", "FLWR_TRAIN_MODE", "FLWR_TARGET_EPSILON", "FLWR_NUM_SERVER_ROUNDS"]:
        os.environ.pop(k, None)


def _sync_with_runtime() -> None:
    """Sincroniza as variáveis de config com .active_config.json ou os.environ se existirem."""
    global DATASET, PARTITIONER, DIRICHLET_ALPHA, TRAIN_MODE, USE_LOCAL_DP_CVAE, TARGET_EPSILON, NUM_SERVER_ROUNDS, EMBEDDING_DIM

    # Prioridade 1: Arquivo ativo em disco
    active_cfg_path = Path("artifacts/.active_config.json")
    if active_cfg_path.exists():
        try:
            data = json.loads(active_cfg_path.read_text(encoding="utf-8"))
            if "DATASET" in data:
                DATASET = data["DATASET"]
            if "PARTITIONER" in data:
                PARTITIONER = data["PARTITIONER"]
            if "DIRICHLET_ALPHA" in data:
                DIRICHLET_ALPHA = float(data["DIRICHLET_ALPHA"])
            if "TRAIN_MODE" in data:
                TRAIN_MODE = data["TRAIN_MODE"]
                USE_LOCAL_DP_CVAE = (TRAIN_MODE == "cvae_dp")
            if "TARGET_EPSILON" in data:
                TARGET_EPSILON = float(data["TARGET_EPSILON"])
            if "NUM_SERVER_ROUNDS" in data:
                NUM_SERVER_ROUNDS = int(data["NUM_SERVER_ROUNDS"])
            return
        except Exception:
            pass

    # Prioridade 2: Variáveis de ambiente
    if "FLWR_DATASET" in os.environ:
        DATASET = os.environ["FLWR_DATASET"]
    if "FLWR_PARTITIONER" in os.environ:
        PARTITIONER = os.environ["FLWR_PARTITIONER"]
    if "FLWR_DIRICHLET_ALPHA" in os.environ:
        DIRICHLET_ALPHA = float(os.environ["FLWR_DIRICHLET_ALPHA"])
    if "FLWR_TRAIN_MODE" in os.environ:
        TRAIN_MODE = os.environ["FLWR_TRAIN_MODE"]
        USE_LOCAL_DP_CVAE = (TRAIN_MODE == "cvae_dp")
    if "FLWR_TARGET_EPSILON" in os.environ:
        TARGET_EPSILON = float(os.environ["FLWR_TARGET_EPSILON"])
    if "FLWR_NUM_SERVER_ROUNDS" in os.environ:
        NUM_SERVER_ROUNDS = int(os.environ["FLWR_NUM_SERVER_ROUNDS"])


_sync_with_runtime()

