
import pandas as pd
import whisper
from pathlib import Path

from config import PROJECT_ROOT

# Load one training record
train_file = PROJECT_ROOT / "data" / "processed" / "splits" / "train.csv"
df = pd.read_csv(train_file)

# Select the first audio file
audio_path = PROJECT_ROOT / df.loc[0, "audio_path"]
reference_text = df.loc[0, "sentence"]

print("Loading Whisper Small model...")
model = whisper.load_model("small", device="cpu")

print("\nTranscribing audio...")
result = model.transcribe(
    str(audio_path),
    language="am",
    fp16=False
)

predicted_text = result["text"].strip()

print("\nREFERENCE TRANSCRIPTION:")
print(reference_text)

print("\nWHISPER PREDICTION:")
print(predicted_text)

print("\nTRANSCRIPTION TEST COMPLETED")
