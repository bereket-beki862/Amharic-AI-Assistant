
import pandas as pd
from jiwer import wer, cer

from config import PROCESSED_DIR

print("WHISPER EVALUATION SUMMARY")
print("=" * 50)

input_file = PROCESSED_DIR / "whisper_evaluation.csv"

if not input_file.exists():
    raise FileNotFoundError(
        "whisper_evaluation.csv not found. "
        "Run whisper_evaluation.py successfully first."
    )

df = pd.read_csv(input_file)

df = df.dropna(subset=["reference", "prediction"])

if df.empty:
    raise ValueError("No valid evaluation samples found.")

references = df["reference"].astype(str).tolist()
predictions = df["prediction"].astype(str).tolist()

word_error_rate = wer(references, predictions)
character_error_rate = cer(references, predictions)

summary = pd.DataFrame([{
    "samples_evaluated": len(df),
    "word_error_rate": word_error_rate,
    "character_error_rate": character_error_rate,
    "approximate_word_accuracy_percent":
        (1 - word_error_rate) * 100
}])

output_file = PROCESSED_DIR / "whisper_evaluation_summary.csv"
summary.to_csv(output_file, index=False)

print("\nEvaluation summary:")
print(summary.round(4).to_string(index=False))

print("\nSummary saved:", output_file)
print("EVALUATION SUMMARY COMPLETED")
