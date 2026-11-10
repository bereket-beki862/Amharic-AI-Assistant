
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Amharic Common Voice dataset directory
DATASET_DIR = PROJECT_ROOT / "data" / "raw"

# Automatically locate the extracted dataset folder
TRAIN_FILE = next(DATASET_DIR.rglob("train.tsv"))

# Actual dataset directory containing train.tsv
DATASET_DIR = TRAIN_FILE.parent

# Audio clips directory
AUDIO_DIR = DATASET_DIR / "clips"

# Print configuration information
print("Dataset connected!")
print("Project root:", PROJECT_ROOT)
print("Dataset location:", DATASET_DIR)
print("Training file exists:", TRAIN_FILE.exists())
print("Audio folder exists:", AUDIO_DIR.exists())
