# ===============================
# Flower / Federated classifier
# ===============================
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
EVAL_GAUSSIAN_NOISE_STD = 0.5
# Dataset partitioning
PARTITIONER = "dirichlet"  # "iid" or "dirichlet"
DIRICHLET_ALPHA = 0.5

# MNIST normalization
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
MEAN = MNIST_MEAN
STD = MNIST_STD

# ===============================
# Important: DP is now applied to the local CVAE, not to the FL classifier.
# The classifier is federated with FedAvg over synthetic embeddings.
# ===============================
USE_LOCAL_DP_CVAE = True

TARGET_DELTA = 1e-5
TARGET_EPSILON = 3.0  # fixed typo: previous file had TARGER_EPSILON
MAX_GRAD_NORM = 1.2
NOISE_MULTIPLIER = 0.8  # kept for compatibility; make_private_with_epsilon calibrates noise

# ===============================
# Frozen MLP embedding extractor
# ===============================
EMBEDDING_DIM = 128
EMBEDDING_HIDDEN_SIZE = 256
EMBEDDING_EXTRACTOR_SEED = 1234
# ===============================
# Public pretraining split for the frozen MLP embedding extractor
# ===============================
# The MNIST train set is split once, with no overlap:
#   - PRETRAIN_RATIO: public/auxiliary subset used to train the embedding extractor
#   - remaining examples: private/federated subset partitioned across clients
# Differential privacy claims in this experiment apply to the private/federated subset.
PRETRAIN_EXTRACTOR = True
PRETRAIN_RATIO = 0.10
DATA_SPLIT_SEED = 2026
PRETRAIN_BATCH_SIZE = 256
PRETRAIN_EPOCHS = 8
PRETRAIN_LR = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-4
FORCE_RETRAIN_EXTRACTOR = False

# Path to the extractor state_dict trained on the public/auxiliary 10%.
# If the file does not exist and PRETRAIN_EXTRACTOR=True, main.py creates it before Flower starts.
EMBEDDING_EXTRACTOR_PATH = "artifacts/mnist_mlp_extractor_public10.pt"

# ===============================
# Local DP-CVAE on embeddings
# ===============================
CVAE_HIDDEN_DIM = 256
CVAE_LATENT_DIM = 32
CVAE_BATCH_SIZE = 256
CVAE_EPOCHS = 10
CVAE_LR = 1e-3
CVAE_BETA = 0.1

# Synthetic dataset generation
SYNTHETIC_MULTIPLIER = 1.0  # 1.0 = generate same number as local real train examples
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
