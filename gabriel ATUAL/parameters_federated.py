USE_DP = True
NUM_PARTITIONS = 10
NUM_SERVER_ROUNDS = 50

TARGET_DELTA = 1e-5  # probabilidade de falha da garantia de privacidade.
MAX_GRAD_NORM = 1.2  #Limiar de clipping C — cada gradiente por amostra é clipado para ter norma L2 ≤ C
NOISE_MULTIPLIER = 0.8  # ruído gaussiano
BATCH_SIZE = 256
EPOCHS = 3

TARGET_EPSILON = 3
FRACTION_FIT=0.1
FRACTION_EVALUATE = 0.1


LR = 0.001
MOMENTUM = 0.9

NUM_CLASSES = 10 # quantas classes de saída no dataset atual

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MEAN = MNIST_MEAN
STD = MNIST_STD