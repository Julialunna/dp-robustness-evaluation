"""
Configurações Globais do Projeto de Aprendizado Federado com DP-CVAE e DINOv2.

Este arquivo é a fonte única de verdade para todos os parâmetros do experimento.
Organizado por seções temáticas para facilitar alterações manuais.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


# ==============================================================================
# 1. DATASET E PARTICIONAMENTO
# ==============================================================================

# Opções de dataset: "pathmnist", "breastmnist", "mnist", etc.
DATASET = "pathmnist"

# Número de classes do dataset
# Exemplo:
# pathmnist = 9
# breastmnist = 2
# mnist = 10
NUM_CLASSES = 9

# Particionamento dos dados entre clientes federados:
# - "dirichlet": não-IID
# - "iid": IID
PARTITIONER = "dirichlet"

# Concentração Dirichlet:
# quanto menor o alpha, maior a heterogeneidade entre clientes
DIRICHLET_ALPHA = 1.0


# ==============================================================================
# 2. EXTRATOR DE FEATURES (DINOv2)
# ==============================================================================

# Modelo extrator
EXTRACTOR_NAME = "dinov2_s"

# Resolução de entrada esperada pelo DINOv2
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
EMBEDDING_DIM = EXTRACTOR_EMBEDDING_DIMS.get(
    EXTRACTOR_NAME,
    384,
)


# ==============================================================================
# 3. CLASSIFICADOR DOWNSTREAM
# ==============================================================================

# Dimensão da camada oculta do EmbeddingClassifier
CLASSIFIER_HIDDEN_DIM = 128


# ==============================================================================
# 4. TREINAMENTO FEDERADO (FLOWER)
# ==============================================================================

# Número total de clientes
NUM_PARTITIONS = 10

# Número total de rodadas federadas
NUM_SERVER_ROUNDS = 20

# Fração de clientes selecionados para treino em cada rodada
FRACTION_FIT = 1.0

# Fração de clientes selecionados para avaliação
FRACTION_EVALUATE = 1.0

# Batch size para classificador local e data loaders
BATCH_SIZE = 256

# Número de épocas locais do classificador
LOCAL_EPOCHS = 1

# Learning rate do classificador
LR = 0.001


# ==============================================================================
# 5. MODALIDADE DE TREINO E DP-CVAE
# ==============================================================================

# Modalidades disponíveis:
#
# "baseline"
#     DINOv2 -> embeddings reais -> classificador
#     Não utiliza CVAE.
#
# "cvae_sem_dp"
#     DINOv2 -> embeddings reais -> CVAE sem DP
#     -> embeddings sintéticos -> classificador
#
# "cvae_dp"
#     DINOv2 -> embeddings reais
#     -> CVAE inteiro treinado com DP-SGD
#        Encoder: DP
#        Decoder: DP
#     -> embeddings sintéticos -> classificador
#
# "cvae_decoder_dp"
#     DINOv2 -> embeddings reais
#     -> CVAE com:
#        Encoder: treinamento normal, sem DP
#        Decoder: treinamento com DP-SGD
#     -> embeddings sintéticos -> classificador
#
TRAIN_MODE = "cvae_decoder_dp"


# ----------------------------------------------------------------------
# FLAGS DERIVADAS DO TRAIN_MODE
# ----------------------------------------------------------------------

# True quando existe algum mecanismo de DP no CVAE
USE_LOCAL_DP_CVAE = TRAIN_MODE in {
    "cvae_dp",
    "cvae_decoder_dp",
}

# True apenas quando DP é aplicada exclusivamente ao decoder
USE_DECODER_ONLY_DP = (
    TRAIN_MODE == "cvae_decoder_dp"
)

# True apenas quando DP é aplicada ao CVAE inteiro
USE_FULL_CVAE_DP = (
    TRAIN_MODE == "cvae_dp"
)


# ----------------------------------------------------------------------
# HIPERPARÂMETROS DE PRIVACIDADE DIFERENCIAL
# ----------------------------------------------------------------------

# Epsilon alvo.
#
# Em:
#   cvae_dp
#       representa epsilon do CVAE inteiro.
#
# Em:
#   cvae_decoder_dp
#       representa epsilon do DECODER.
#
TARGET_EPSILON = 5.0

# Delta utilizado pelo accountant do Opacus
TARGET_DELTA = 1e-5

# Norma máxima utilizada no clipping DP.
#
# Em cvae_decoder_dp, este valor será aplicado
# SOMENTE aos gradientes do decoder.
MAX_GRAD_NORM = 1.2


# ----------------------------------------------------------------------
# HIPERPARÂMETROS DO CVAE
# ----------------------------------------------------------------------

CVAE_HIDDEN_DIM = 256

CVAE_LATENT_DIM = 32

CVAE_BATCH_SIZE = 256

CVAE_EPOCHS = 10

CVAE_LR = 0.001

# Peso da divergência KL
CVAE_BETA = 0.1


# ==============================================================================
# 6. CACHE E REPRODUTIBILIDADE
# ==============================================================================

REAL_EMBEDDINGS_CACHE_DIR = (
    "artifacts/real_embeddings"
)

SYNTHETIC_CACHE_DIR = (
    "artifacts/synthetic_embeddings"
)

TEST_EMBEDDINGS_CACHE_DIR = (
    "artifacts/test_embeddings"
)

# Alterado de 4 para 5 porque o método de geração
# mudou e não queremos reutilizar caches antigos
SYNTHETIC_CACHE_VERSION = 5


# Se True, ignora o cache sintético existente
FORCE_REGENERATE_SYNTHETICS = False

# Se False, o CVAE/decoder é treinado apenas quando
# o conjunto sintético precisa ser criado
RETRAIN_CVAE_EVERY_ROUND = False


# ==============================================================================
# 7. AVALIAÇÃO DE ROBUSTEZ A RUÍDO
# ==============================================================================

# Seed para geração determinística de ruído
EVAL_GAUSSIAN_NOISE_SEED = 1234

# Níveis de ruído gaussiano adicionados às imagens
EVAL_GAUSSIAN_NOISE_STD1 = 0.08
EVAL_GAUSSIAN_NOISE_STD2 = 0.16
EVAL_GAUSSIAN_NOISE_STD3 = 0.26


# ==============================================================================
# 8. SIMULAÇÃO E EXPERIMENTOS
# ==============================================================================

# Número de repetições do experimento
NUM_RUNS = 5

# Seeds utilizadas nos experimentos
GRID_SEEDS = [
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2032,
    2033,
    2034,
    2035,
]

BASE_SEED = 2026


# ==============================================================================
# 9. FUNÇÕES AUXILIARES PARA FLAGS DO TRAIN_MODE
# ==============================================================================

def _update_train_mode_flags() -> None:
    """
    Atualiza automaticamente as flags derivadas de TRAIN_MODE.

    Isso é necessário porque TRAIN_MODE pode ser alterado
    dinamicamente pelo run_experiments.py ou por workers Ray.
    """
    global USE_LOCAL_DP_CVAE
    global USE_DECODER_ONLY_DP
    global USE_FULL_CVAE_DP

    USE_LOCAL_DP_CVAE = TRAIN_MODE in {
        "cvae_dp",
        "cvae_decoder_dp",
    }

    USE_DECODER_ONLY_DP = (
        TRAIN_MODE == "cvae_decoder_dp"
    )

    USE_FULL_CVAE_DP = (
        TRAIN_MODE == "cvae_dp"
    )


# ==============================================================================
# 10. SINCRONIZAÇÃO DINÂMICA ENTRE PROCESSOS
# ==============================================================================

def get_env_vars() -> Dict[str, str]:
    """
    Retorna as configurações que precisam ser propagadas
    para subprocessos/Workers Ray.
    """

    return {
        "FLWR_DATASET": str(DATASET),
        "FLWR_PARTITIONER": str(PARTITIONER),
        "FLWR_DIRICHLET_ALPHA": str(
            DIRICHLET_ALPHA
        ),
        "FLWR_TRAIN_MODE": str(TRAIN_MODE),
        "FLWR_TARGET_EPSILON": str(
            TARGET_EPSILON
        ),
        "FLWR_NUM_SERVER_ROUNDS": str(
            NUM_SERVER_ROUNDS
        ),
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
    Aplica configuração dinâmica no processo atual
    e propaga para subprocessos/workers Ray.
    """

    global DATASET
    global PARTITIONER
    global DIRICHLET_ALPHA
    global TRAIN_MODE
    global TARGET_EPSILON
    global NUM_SERVER_ROUNDS
    global EMBEDDING_DIM

    if dataset is not None:
        DATASET = dataset

    if partitioner is not None:
        PARTITIONER = partitioner

    if dirichlet_alpha is not None:
        DIRICHLET_ALPHA = float(
            dirichlet_alpha
        )

    if train_mode is not None:
        TRAIN_MODE = train_mode

        # Sempre recalcula as flags derivadas
        _update_train_mode_flags()

    if target_epsilon is not None:
        TARGET_EPSILON = float(
            target_epsilon
        )

    if num_server_rounds is not None:
        NUM_SERVER_ROUNDS = int(
            num_server_rounds
        )

    EMBEDDING_DIM = (
        EXTRACTOR_EMBEDDING_DIMS.get(
            EXTRACTOR_NAME,
            384,
        )
    )

    # ------------------------------------------------------------------
    # 1. Variáveis de ambiente
    # ------------------------------------------------------------------

    for key, value in get_env_vars().items():
        os.environ[key] = value

    # ------------------------------------------------------------------
    # 2. Arquivo persistente para subprocessos isolados
    # ------------------------------------------------------------------

    active_cfg_path = Path(
        "artifacts/.active_config.json"
    )

    try:
        active_cfg_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        active_cfg_path.write_text(
            json.dumps(
                {
                    "DATASET": DATASET,
                    "PARTITIONER": PARTITIONER,
                    "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
                    "TRAIN_MODE": TRAIN_MODE,

                    "USE_LOCAL_DP_CVAE":
                        USE_LOCAL_DP_CVAE,

                    "USE_DECODER_ONLY_DP":
                        USE_DECODER_ONLY_DP,

                    "USE_FULL_CVAE_DP":
                        USE_FULL_CVAE_DP,

                    "TARGET_EPSILON":
                        TARGET_EPSILON,

                    "NUM_SERVER_ROUNDS":
                        NUM_SERVER_ROUNDS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    except Exception:
        pass


def clear_active_config() -> None:
    """
    Remove configuração ativa após o término
    da simulação.
    """

    active_cfg_path = Path(
        "artifacts/.active_config.json"
    )

    if active_cfg_path.exists():
        try:
            active_cfg_path.unlink()
        except Exception:
            pass

    env_keys = [
        "FLWR_DATASET",
        "FLWR_PARTITIONER",
        "FLWR_DIRICHLET_ALPHA",
        "FLWR_TRAIN_MODE",
        "FLWR_TARGET_EPSILON",
        "FLWR_NUM_SERVER_ROUNDS",
    ]

    for key in env_keys:
        os.environ.pop(
            key,
            None,
        )


def _sync_with_runtime() -> None:
    """
    Sincroniza a configuração com:

    1. artifacts/.active_config.json
    2. variáveis de ambiente

    Isso é importante para workers Ray/Flower.
    """

    global DATASET
    global PARTITIONER
    global DIRICHLET_ALPHA
    global TRAIN_MODE
    global TARGET_EPSILON
    global NUM_SERVER_ROUNDS
    global EMBEDDING_DIM

    # ------------------------------------------------------------------
    # PRIORIDADE 1: arquivo ativo
    # ------------------------------------------------------------------

    active_cfg_path = Path(
        "artifacts/.active_config.json"
    )

    if active_cfg_path.exists():
        try:
            data = json.loads(
                active_cfg_path.read_text(
                    encoding="utf-8"
                )
            )

            if "DATASET" in data:
                DATASET = data["DATASET"]

            if "PARTITIONER" in data:
                PARTITIONER = data[
                    "PARTITIONER"
                ]

            if "DIRICHLET_ALPHA" in data:
                DIRICHLET_ALPHA = float(
                    data[
                        "DIRICHLET_ALPHA"
                    ]
                )

            if "TRAIN_MODE" in data:
                TRAIN_MODE = data[
                    "TRAIN_MODE"
                ]

                _update_train_mode_flags()

            if "TARGET_EPSILON" in data:
                TARGET_EPSILON = float(
                    data[
                        "TARGET_EPSILON"
                    ]
                )

            if "NUM_SERVER_ROUNDS" in data:
                NUM_SERVER_ROUNDS = int(
                    data[
                        "NUM_SERVER_ROUNDS"
                    ]
                )

            EMBEDDING_DIM = (
                EXTRACTOR_EMBEDDING_DIMS.get(
                    EXTRACTOR_NAME,
                    384,
                )
            )

            return

        except Exception:
            pass

    # ------------------------------------------------------------------
    # PRIORIDADE 2: variáveis de ambiente
    # ------------------------------------------------------------------

    if "FLWR_DATASET" in os.environ:
        DATASET = os.environ[
            "FLWR_DATASET"
        ]

    if "FLWR_PARTITIONER" in os.environ:
        PARTITIONER = os.environ[
            "FLWR_PARTITIONER"
        ]

    if "FLWR_DIRICHLET_ALPHA" in os.environ:
        DIRICHLET_ALPHA = float(
            os.environ[
                "FLWR_DIRICHLET_ALPHA"
            ]
        )

    if "FLWR_TRAIN_MODE" in os.environ:
        TRAIN_MODE = os.environ[
            "FLWR_TRAIN_MODE"
        ]

        _update_train_mode_flags()

    if "FLWR_TARGET_EPSILON" in os.environ:
        TARGET_EPSILON = float(
            os.environ[
                "FLWR_TARGET_EPSILON"
            ]
        )

    if "FLWR_NUM_SERVER_ROUNDS" in os.environ:
        NUM_SERVER_ROUNDS = int(
            os.environ[
                "FLWR_NUM_SERVER_ROUNDS"
            ]
        )

    EMBEDDING_DIM = (
        EXTRACTOR_EMBEDDING_DIMS.get(
            EXTRACTOR_NAME,
            384,
        )
    )


# ==============================================================================
# 11. SINCRONIZA CONFIGURAÇÃO NO IMPORT
# ==============================================================================

_sync_with_runtime()