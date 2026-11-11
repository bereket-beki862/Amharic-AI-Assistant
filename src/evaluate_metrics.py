
import pandas as pd
import jiwer

from config import PROJECT_ROOT

# Load Whisper evaluation results
input_file = PROJECT_ROOT / "data" / "processed" / "whisper_evaluation.csv"
df = pd.read_csv(input_file)

# Calculate WER and CER
references = df["reference"].astype(str).tolist()
predictions = df["prediction"].astype(str).tolist()

wer = jiwer.wer(references, predictions)
cer = jiwer.cer(references, predictions)

print("WHISPER EVALUATION RESULTS")
print("--------------------------")
print("Samples evaluated:", len(df))
print("Word Error Rate (WER):", round(wer, 4))
print("Character Error Rate (CER):", round(cer, 4))
print("Word Accuracy (approx.):", round(max(0, 1 - wer) * 100, 2), "%")
