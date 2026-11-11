
import pandas as pd
from pathlib import Path

from config import TRAIN_FILE, AUDIO_DIR

# Load dataset
df = pd.read_csv(TRAIN_FILE, sep="\t")

print("\n========== AMHARIC DATASET INSPECTION ==========\n")

# 1. Dataset information
print("Total records:", len(df))
print("Total columns:", len(df.columns))

# 2. Required columns
required_columns = ["path", "sentence"]

for column in required_columns:
    if column in df.columns:
        print(f"{column}: Found")
    else:
        print(f"{column}: Missing")

# 3. Missing values
print("\nMissing values:")
print(df[required_columns].isnull().sum())

# 4. Duplicate records
print("\nDuplicate audio paths:", df["path"].duplicated().sum())
print("Duplicate sentences:", df["sentence"].duplicated().sum())

# 5. Sentence length
df["sentence_length"] = df["sentence"].astype(str).str.len()

print("\nSentence length statistics:")
print(df["sentence_length"].describe())

# 6. Audio file verification
missing_audio = []

for audio_file in df["path"]:
    if not (AUDIO_DIR / str(audio_file)).is_file():
        missing_audio.append(audio_file)

print("\nAudio verification:")
print("Available audio files:", len(df) - len(missing_audio))
print("Missing audio files:", len(missing_audio))

# 7. Show sample sentences
print("\nFirst 5 Amharic sentences:")

for index, sentence in enumerate(df["sentence"].head(5), start=1):
    print(f"{index}. {sentence}")

print("\n========== INSPECTION COMPLETED ==========\n")
