
import pandas as pd

from config import AUDIO_DIR, PROCESSED_DIR, PROJECT_ROOT

print("AMHARIC DATASET PREPARATION")
print("=" * 50)

# Load cleaned transcription dataset
input_file = PROCESSED_DIR / "cleaned_train.csv"
df = pd.read_csv(input_file)

print("Original records:", len(df))

# Check required columns
required_columns = ["path", "sentence"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

# Remove invalid transcription entries
df = df.dropna(subset=["path", "sentence"]).copy()

df["path"] = df["path"].astype(str).str.strip()
df["sentence"] = df["sentence"].astype(str).str.strip()

df = df[
    (df["path"] != "") &
    (df["sentence"] != "")
].copy()

# Check audio files
audio_paths = []
missing_files = []

for filename in df["path"]:
    audio_file = AUDIO_DIR / filename

    if audio_file.is_file():
        relative_path = audio_file.relative_to(PROJECT_ROOT)
        audio_paths.append(relative_path.as_posix())
    else:
        audio_paths.append(None)
        missing_files.append(filename)

df["audio_path"] = audio_paths

print("Valid transcription records:", len(df))
print("Missing audio files:", len(missing_files))

# Stop if audio files are missing
if missing_files:
    print("\nMissing audio examples:")
    for filename in missing_files[:10]:
        print(filename)

    raise FileNotFoundError(
        "Some audio files are missing. Please check the dataset."
    )

# Keep only required columns
df = df[["path", "sentence", "audio_path"]]

# Shuffle records reproducibly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into training, validation and testing datasets
total_records = len(df)

train_end = int(total_records * 0.80)
validation_end = train_end + int(total_records * 0.10)

train_df = df.iloc[:train_end]
validation_df = df.iloc[train_end:validation_end]
test_df = df.iloc[validation_end:]

# Create output directory
split_dir = PROCESSED_DIR / "splits"
split_dir.mkdir(parents=True, exist_ok=True)

# Save datasets
train_df.to_csv(split_dir / "train.csv", index=False)
validation_df.to_csv(split_dir / "validation.csv", index=False)
test_df.to_csv(split_dir / "test.csv", index=False)

print("\nDATASET SPLIT RESULTS")
print("=" * 50)
print("Total records:", total_records)
print("Training records:", len(train_df))
print("Validation records:", len(validation_df))
print("Testing records:", len(test_df))

print("\nSaved files:")
print(split_dir / "train.csv")
print(split_dir / "validation.csv")
print(split_dir / "test.csv")

print("\nDATASET PREPARATION COMPLETED")
