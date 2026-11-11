
import pandas as pd
import librosa
from pathlib import Path
import numpy as np

from config import AUDIO_DIR, PROCESSED_DIR

# Load cleaned dataset
input_file = PROCESSED_DIR / "cleaned_train.csv"
df = pd.read_csv(input_file)

print("AMHARIC AUDIO ANALYSIS")
print("=" * 50)

durations = []
sample_rates = []
channels = []
missing_files = []

# Analyze audio files
for index, row in df.iterrows():

    audio_path = AUDIO_DIR / row["path"]

    if not audio_path.exists():
        missing_files.append(row["path"])
        continue

    try:
        audio, sample_rate = librosa.load(
            audio_path,
            sr=None,
            mono=False
        )

        duration = librosa.get_duration(
            y=audio,
            sr=sample_rate
        )

        durations.append(duration)
        sample_rates.append(sample_rate)

        if audio.ndim == 1:
            channels.append(1)
        else:
            channels.append(audio.shape[0])

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} files")

    except Exception as error:
        print(f"Error in {row['path']}: {error}")

print("\nANALYSIS RESULTS")
print("=" * 50)

print("Total dataset records:", len(df))
print("Successfully analyzed:", len(durations))
print("Missing files:", len(missing_files))

if durations:
    print("\nAudio duration (seconds):")
    print("Minimum:", round(min(durations), 2))
    print("Maximum:", round(max(durations), 2))
    print("Average:", round(float(np.mean(durations)), 2))

    print("\nSample rates:")
    print(pd.Series(sample_rates).value_counts())

    print("\nAudio channels:")
    print(pd.Series(channels).value_counts())

print("\nAUDIO ANALYSIS COMPLETED")
