from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# dataset
DATASET_FLAG = "pneumoniamnist"
IMAGE_SIZE = 224
SPLIT = "test"
MAX_IMAGES = 40

# general design
NOISE_TYPES = ["gaussian", "poisson"]
MODES = ["rgb", "gray"]
METHODS = ["bm3d", "wavelet"]

# noises
GAUSSIAN_SIGMA = 0.08
POISSON_PEAK = 40.0

# Wavelet
WAVELET_NAME = "db2"
WAVELET_METHOD = "BayesShrink"
WAVELET_MODE = "soft"

# BM3D
BM3D_SIGMA_GAUSSIAN = GAUSSIAN_SIGMA
BM3D_SIGMA_ANSCOMBE = 1.0

# reproducibility
RANDOM_SEED = 42

# output
EXAMPLE_ROWS = 4
SAVE_DPI = 220

#plot
COLOR_BM3D = "#4E79A7"
COLOR_WAVELET = "#59A14F"
COLOR_GRID = "#D9D9D9"
COLOR_HEADER = "#222222"