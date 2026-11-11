
import pandas as pd
from pathlib import Path
import re

from config import TRAIN_FILE, AUDIO_DIR, PROJECT_ROOT

# Load original dataset
df = pd.read_csv(TRAIN_FILE, sep="\t")

print("\n========== DATASET PREPROCESSING ==========\n")

print("Original records:", len(df))

# 1. Keep only the required columns
df = df[["path", "sentence"]].copy()

# 2. Remove leading and trailing whitespace
df["sentence"] = df["sentence"].astype(str).str.strip()

# 3. Normalize multiple spaces into one space
df["sentence"] = df["sentence"].apply(
    lambda text: re.sub(r"\s+", " ", text)
)

# 4. Remove empty or invalid sentences
df = df[
    df["sentence"].notna() &
    (df["sentence"] != "") &
    (df["sentence"].str.lower() != "nan")
].copy()

# 5. Check that audio files exist
df["audio_exists"] = df["path"].apply(
    lambda path: (AUDIO_DIR / str(path)).is_file()
)

missing_audio = (~df["audio_exists"]).sum()

# Keep only records with existing audio
df = df[df["audio_exists"]].copy()

# 6. Remove duplicate audio paths
df = df.drop_duplicates(subset=["path"])

# Remove helper column
df = df.drop(columns=["audio_exists"])

# 7. Save cleaned dataset
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "cleaned_train.csv"

df.to_csv(output_file, index=False, encoding="utf-8")

# 8. Display results
print("Missing audio files:", missing_audio)
print("Cleaned records:", len(df))
print("Removed records:", 563 - len(df))

print("\nFirst 5 cleaned sentences:")
for index, sentence in enumerate(df["sentence"].head(5), start=1):
    print(f"{index}. {sentence}")

print("\nSaved cleaned dataset to:")
print(output_file)

print("\n========== PREPROCESSING COMPLETED ==========\n")
