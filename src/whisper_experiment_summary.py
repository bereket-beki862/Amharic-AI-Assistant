import pandas as pd
from pathlib import Path

from config import PROJECT_ROOT

# Load experiment metrics
metrics_file = PROJECT_ROOT / "data" / "processed" / "whisper_experiment_metrics.csv"

df = pd.read_csv(metrics_file)

# Calculate summary
baseline_wer = df["baseline_wer"].mean()
experiment_wer = df["experiment_wer"].mean()

baseline_cer = df["baseline_cer"].mean()
experiment_cer = df["experiment_cer"].mean()

wer_improvement = baseline_wer - experiment_wer
cer_improvement = baseline_cer - experiment_cer

# Display summary
print("\nWHISPER EXPERIMENT SUMMARY")
print("=" * 45)

print(f"Samples evaluated: {len(df)}")

print(f"\nBaseline WER:   {baseline_wer:.4f}")
print(f"Experiment WER:  {experiment_wer:.4f}")
print(f"WER improvement: {wer_improvement:.4f}")

print(f"\nBaseline CER:    {baseline_cer:.4f}")
print(f"Experiment CER:  {experiment_cer:.4f}")
print(f"CER improvement: {cer_improvement:.4f}")

if wer_improvement > 0 and cer_improvement > 0:
    print("\nRESULT: Experiment improved recognition.")
elif wer_improvement < 0 and cer_improvement < 0:
    print("\nRESULT: Experiment worsened recognition.")
else:
    print("\nRESULT: Mixed results or no clear improvement.")

# Save summary
summary = pd.DataFrame([{
    "samples_evaluated": len(df),
    "baseline_wer": baseline_wer,
    "experiment_wer": experiment_wer,
    "wer_improvement": wer_improvement,
    "baseline_cer": baseline_cer,
    "experiment_cer": experiment_cer,
    "cer_improvement": cer_improvement
}])

output_file = PROJECT_ROOT / "data" / "processed" / "whisper_experiment_summary.csv"
summary.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nSummary saved to:", output_file)