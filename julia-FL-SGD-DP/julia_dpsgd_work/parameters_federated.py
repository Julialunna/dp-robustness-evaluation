
NUM_PARTITIONS = 10
NUM_SERVER_ROUNDS = 20
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0
USE_LOCAL_DP_CVAE = False
BATCH_SIZE = 256
EPOCHS = 3  # local epochs of the federated downstream classifier
LR = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 10
EVAL_GAUSSIAN_NOISE_SEED = 1234
EVAL_GAUSSIAN_NOISE_STD1 = 0.1
EVAL_GAUSSIAN_NOISE_STD2 = 0.25  
EVAL_GAUSSIAN_NOISE_STD3 = 0.5
# Dataset partitioning
PARTITIONER = "iid"  #  or "dirichlet"
DIRICHLET_ALPHA = 1
FOUNDATION_MODEL = "efficientnet_b0"
FOUNDATION_IMAGE_SIZE = 224
EMBEDDING_DIM = 1280
EMBEDDING_HIDDEN_SIZE = 256
# # MobileNetV3-Small
# FOUNDATION_MODEL = "mobilenet_v3_small"
# EMBEDDING_DIM = 576

# # DINOv2-S
# FOUNDATION_MODEL = "dinov2_s"
# EMBEDDING_DIM = 384

# MNIST normalization
#dados das imagens (varia de acordo com dataset)
IMG_SIZE = 28
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MEAN = MNIST_MEAN
STD = MNIST_STD

TARGET_DELTA = 1e-5
TARGET_EPSILON = 3.0  # fixed typo: previous file had TARGER_EPSILON
MAX_GRAD_NORM = 1.2
NOISE_MULTIPLIER = 0.8  # kept for compatibility; make_private_with_epsilon calibrates noise

EMBEDDING_DIM = 128
#e usar EfficientNet-B0 com embedding 1280 ou MedMNIST mais difícil colocar 512
EMBEDDING_HIDDEN_SIZE = 256
EMBEDDING_EXTRACTOR_SEED = 1234

PRETRAIN_EXTRACTOR = True
PRETRAIN_RATIO = 0.10
DATA_SPLIT_SEED = 2026
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_EPOCHS = 8
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-4
FORCE_RETRAIN_EXTRACTOR = False


EMBEDDING_EXTRACTOR_PATH = "artifacts/mnist_mlp_extractor_public10.pt"


CVAE_HIDDEN_DIM = 256
CVAE_LATENT_DIM = 32
CVAE_BATCH_SIZE = 128
CVAE_EPOCHS = 20
CVAE_LR = 1e-3
CVAE_BETA = 0.1

# Synthetic dataset generation
SYNTHETIC_MULTIPLIER = 1.0  # 1.0 gera a mesma quantidade de imagens que o cliente tem, podemos talvez tentar ter 2.0 ver se melhora a performance, mas aumenta o tempo de execução. 0.5 gera metade das imagens do cliente, 2.0 gera o dobro.
# "uniform" avoids directly reusing exact private label counts; "real_distribution" may improve non-IID utility.
SYNTHETIC_LABEL_MODE = "real_distribution"
SYNTHETIC_CACHE_DIR = "artifacts/synthetic_embeddings"
# Increment this whenever the synthetic-generation pipeline changes.
# Old cache files with a different version will be ignored and regenerated.
SYNTHETIC_CACHE_VERSION = 2
# If True, ignore any saved synthetic embeddings and regenerate the local DP-CVAE cache.
# Keep False after the first corrected run to avoid spending time retraining every execution.
FORCE_REGENERATE_SYNTHETICS = False
RETRAIN_CVAE_EVERY_ROUND = False
