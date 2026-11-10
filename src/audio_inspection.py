
import pandas as pd
import librosa

from config import TRAIN_FILE, AUDIO_DIR

# Load training dataset
df = pd.read_csv(TRAIN_FILE, sep="\t")

print("Checking first 5 audio recordings...\n")

for index, row in df.head(5).iterrows():
    audio_path = AUDIO_DIR / row["path"]

    try:
        audio, sample_rate = librosa.load(audio_path, sr=None)

        duration = len(audio) / sample_rate

        print(f"Recording {index + 1}")
        print("Sample rate:", sample_rate)
        print("Duration:", round(duration, 2), "seconds")
        print("Transcription:", row["sentence"])
        print("Status: Successfully loaded")
        print("-" * 40)

    except Exception as error:
        print("Error:", audio_path)
        print(error)
