
import pandas as pd
from jiwer import wer, cer

from config import PROCESSED_DIR

print("WHISPER TRANSCRIPTION QUALITY ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "whisper_evaluation.csv"

df = pd.read_csv(input_file)

required_columns = ["reference", "prediction"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

df = df.dropna(subset=required_columns)

references = df["reference"].astype(str).tolist()
predictions = df["prediction"].astype(str).tolist()

if not references:
    raise ValueError("No valid evaluation samples found.")

word_error = wer(references, predictions)
character_error = cer(references, predictions)

print("\nSamples evaluated:", len(references))
print("Word Error Rate:", round(word_error, 4))
print("Character Error Rate:", round(character_error, 4))
print("Approximate word accuracy:", round((1 - word_error) * 100, 2), "%")

print("\nTRANSCRIPTION QUALITY ANALYSIS COMPLETED")
