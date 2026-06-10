# ===============================
# Flower / Federated classifier
# ===============================
NUM_PARTITIONS = 10
NUM_SERVER_ROUNDS = 20
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0

BATCH_SIZE = 256
EPOCHS = 3  # local epochs of the federated downstream classifier
LR = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 10

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
# Optional: path to a pretrained public/DP-trained extractor state_dict.
# If None, all clients/server use the same deterministic frozen random MLP.
EMBEDDING_EXTRACTOR_PATH = None

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
SYNTHETIC_LABEL_MODE = "uniform"
SYNTHETIC_CACHE_DIR = "artifacts/synthetic_embeddings"
RETRAIN_CVAE_EVERY_ROUND = False
