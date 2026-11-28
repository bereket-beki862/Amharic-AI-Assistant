import pandas as pd
from jiwer import wer, cer
from config import PROJECT_ROOT

# Load normalized evaluation results
input_file = (
    PROJECT_ROOT / "data" / "processed" / "whisper_normalized_evaluation.csv"
)

df = pd.read_csv(input_file, encoding="utf-8-sig")

results = []

print("=" * 50)
print("AMHARIC NORMALIZED WER AND CER EVALUATION")
print("=" * 50)

for i, row in df.iterrows():
    reference = str(row["normalized_reference"])
    prediction = str(row["normalized_prediction"])

    sample_wer = wer(reference, prediction)
    sample_cer = cer(reference, prediction)

    results.append({
        "sample": i + 1,
        "normalized_wer": sample_wer,
        "normalized_cer": sample_cer
    })

    print(f"\nSample {i + 1}")
    print(f"WER: {sample_wer:.4f}")
    print(f"CER: {sample_cer:.4f}")

# Create metrics DataFrame
metrics_df = pd.DataFrame(results)

# Calculate averages
avg_wer = metrics_df["normalized_wer"].mean()
avg_cer = metrics_df["normalized_cer"].mean()

print("\n" + "=" * 50)
print("OVERALL RESULTS")
print("=" * 50)

print(f"Average WER: {avg_wer:.4f}")
print(f"Average CER: {avg_cer:.4f}")

# Save results
output_file = (
    PROJECT_ROOT / "data" / "processed" / "whisper_amharic_normalized_metrics.csv"
)

metrics_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nMetrics saved to:")
print(output_file)
print("\nEVALUATION COMPLETED")