
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import AUDIO_DIR, PROCESSED_DIR
from config import TRAIN_FILE

print("AMHARIC AUDIO FEATURE EXTRACTION")
print("=" * 50)

# Load the cleaned dataset
cleaned_file = PROCESSED_DIR / "cleaned_train.csv"

if not cleaned_file.exists():
    raise FileNotFoundError(f"Missing file: {cleaned_file}")

df = pd.read_csv(cleaned_file)

features_list = []

for index, row in tqdm(df.iterrows(), total=len(df)):

    audio_path = AUDIO_DIR / row["path"]

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=32000)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13
        )
        mfcc_mean = np.mean(mfcc, axis=1)

        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr
        )
        centroid_mean = np.mean(spectral_centroid)

        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Audio duration
        duration = librosa.get_duration(y=y, sr=sr)

        feature_row = {
            "path": row["path"],
            "sentence": row["sentence"],
            "duration": duration,
            "spectral_centroid": centroid_mean,
            "zero_crossing_rate": zcr_mean
        }

        for i, value in enumerate(mfcc_mean):
            feature_row[f"mfcc_{i+1}"] = value

        for i, value in enumerate(chroma_mean):
            feature_row[f"chroma_{i+1}"] = value

        features_list.append(feature_row)

    except Exception as error:
        print(f"\nError processing {audio_path}: {error}")

# Save extracted features
features_df = pd.DataFrame(features_list)

output_file = PROCESSED_DIR / "audio_features.csv"
features_df.to_csv(output_file, index=False)

print("\nFEATURE EXTRACTION RESULTS")
print("=" * 50)
print("Total records:", len(df))
print("Successfully extracted:", len(features_df))
print("Failed records:", len(df) - len(features_df))
print("Feature columns:", len(features_df.columns))
print("Saved to:", output_file)
print("\nFEATURE EXTRACTION COMPLETED")
