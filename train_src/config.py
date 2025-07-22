# Configuration parameters for MA-VAE

# Data parameters
N_FEATURES = 129  # Original number of features
REDUCED_FEATURES = 129  # Can be modified when applying dimension reduction
SEQUENCE_LENGTH = 58  # Maximum sequence length in the dataset
WINDOW_SIZE = 32  # Size of sliding window (about 1/2 of min sequence length)
WINDOW_SHIFT = 16  # Shift size for sliding window (2/3 overlap for more samples)

# Model parameters
LATENT_DIM = 64  # Dimension of latent space
BATCH_SIZE = 32  # Batch size for training
# LSTM_UNITS = [192]  # Single BiLSTM layer (~1.5x input features)
LSTM_UNITS = [256, 128]  # Single BiLSTM layer (~1.5x input features)

ATTENTION_HEADS = 8  # Number of attention heads
ATTENTION_KEY_DIM = 64  # Key dimension in attention mechanism

# Training parameters
INITIAL_BETA = 1e-8  # Initial KL weight
ANNEALING_EPOCHS = 25  # Number of epochs for KL annealing
GRACE_PERIOD = 25  # Grace period for annealing
MAX_EPOCHS = 10000  # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = 250  # Patience for early stopping

# Random seed
RANDOM_SEED = 1 