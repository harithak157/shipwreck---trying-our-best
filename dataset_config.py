from pathlib import Path

# Root directory where the dataset will be extracted/mounted.
DATA_ROOT = Path("data/raw")

IMAGE_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"

PROCESSED_DIR = Path("data/processed")

IMAGE_SIZE = 512

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
