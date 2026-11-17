
import pandas as pd
from jiwer import wer

from config import PROCESSED_DIR

print("WHISPER TRANSCRIPTION ERROR ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "whisper_evaluation.csv"

if not input_file.exists():
    raise FileNotFoundError(
        "whisper_evaluation.csv not found. "
        "Run whisper_evaluation.py successfully first."
    )

df = pd.read_csv(input_file)

required = ["audio_path", "reference", "prediction"]

if not all(column in df.columns for column in required):
    raise ValueError("Missing required evaluation columns.")

df = df.dropna(subset=["reference", "prediction"]).copy()

df["sample_wer"] = df.apply(
    lambda row: wer(
        [str(row["reference"])],
        [str(row["prediction"])]
    ),
    axis=1
)

df = df.sort_values("sample_wer", ascending=False)

output_file = PROCESSED_DIR / "whisper_error_analysis.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nHighest-error samples:")
print(
    df[["audio_path", "sample_wer"]]
    .head(10)
    .to_string(index=False)
)

print("\nReport saved:", output_file)
print("ERROR ANALYSIS COMPLETED")
