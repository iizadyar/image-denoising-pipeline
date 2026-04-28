from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / "checkpoints"

# Dataset
DATASET_FLAG = "pneumoniamnist"
IMAGE_SIZE = 224

# Classical pipeline
SPLIT = "test"
MAX_IMAGES = 40
NOISE_TYPES = ["gaussian", "poisson"]
MODES = ["gray", "rgb"]
METHODS = ["bm3d", "wavelet"]

# Noise settings
GAUSSIAN_SIGMA = 0.08
POISSON_PEAK = 40.0

# Wavelet settings
WAVELET_NAME = "db2"
WAVELET_METHOD = "BayesShrink"
WAVELET_MODE = "soft"

# BM3D settings
BM3D_SIGMA_GAUSSIAN = GAUSSIAN_SIGMA
BM3D_SIGMA_ANSCOMBE = 1.0

# DnCNN settings
DN_TRAIN_NOISE_TYPES = ["gaussian", "poisson"]
DN_MODE = "gray"
DN_BATCH_SIZE = 8
DN_EPOCHS = 10
DN_LR = 1e-3
DN_NUM_WORKERS = 2
DN_SAVE_NAME = "dncnn_best.pth"

# Reproducibility
RANDOM_SEED = 42

# Visualization
EXAMPLE_ROWS = 4
SAVE_DPI = 240

METHOD_COLORS = {
    "bm3d": "#4C78A8",
    "wavelet": "#54A24B",
    "dncnn": "#E45756",
}