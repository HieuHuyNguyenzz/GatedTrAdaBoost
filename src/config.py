import torch

# General Settings
NUM_FEATURE = 256
NUM_CLASSES = 3
PACKET_NUM = 20
CLIENT_LR = 1e-4
NUM_EPOCHS = 30
BATCH_SIZE = 64  # Increased for better GPU utilization on M-series
SEED = 42

# DataLoader Settings for Apple Silicon
NUM_WORKERS = 4  # Parallel data loading for faster preprocessing

# Gated AdaBoost Settings
GATING_K = 3 # Top-k learners to select
GATING_TAU = 1.0 # Temperature for softmax
GATING_LR = 1e-4
GATING_EPOCHS = 20
LAMBDA_KL = 1.0
LAMBDA_RANK = 0.1
LAMBDA_SPARSE = 0.1
LAMBDA_MARGIN = 1.0

# Device Configuration for MacBook M-series and others
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Data Paths (Can be overridden by environment variables)
DATA_DIR = "SOICT Data"
SAME_DIST_PATH = f"{DATA_DIR}/df0.01.feather"
DIFF_1_DIST_PATH = f"{DATA_DIR}/A_data1.feather"
TEST_DIST_PATH = f"{DATA_DIR}/df0.99.feather"
