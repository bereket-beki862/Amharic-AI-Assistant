import pandas as pd
from jiwer import wer, cer
from pathlib import Path

from config import PROJECT_ROOT

# Existing result files
baseline_file = PROJECT_ROOT / "data" / "processed" / "whisper_evaluation.csv"
experiment_file = PROJECT_ROOT / "data" / "processed" / "whisper_decoding_experiment.csv"

# Load CSV files
baseline_df = pd.read_csv(baseline_file)
experiment_df = pd.read_csv(experiment_file)

results = []

# Compare baseline and experiment predictions
for i in range(min(len(baseline_df), len(experiment_df))):

    reference = str(baseline_df.loc[i, "reference"])
    baseline_prediction = str(baseline_df.loc[i, "prediction"])
    experiment_prediction = str(experiment_df.loc[i, "prediction"])

    results.append({
        "sample": i + 1,
        "baseline_wer": wer(reference, baseline_prediction),
        "experiment_wer": wer(reference, experiment_prediction),
        "baseline_cer": cer(reference, baseline_prediction),
        "experiment_cer": cer(reference, experiment_prediction)
    })

metrics_df = pd.DataFrame(results)

# Save report
output_file = PROJECT_ROOT / "data" / "processed" / "whisper_experiment_metrics.csv"
metrics_df.to_csv(output_file, index=False, encoding="utf-8-sig")

# Display summary
print("\nWHISPER EXPERIMENT METRICS")
print("=" * 40)
print(metrics_df.to_string(index=False))

print("\nAVERAGES")
print("Baseline WER:", metrics_df["baseline_wer"].mean())
print("Experiment WER:", metrics_df["experiment_wer"].mean())
print("Baseline CER:", metrics_df["baseline_cer"].mean())
print("Experiment CER:", metrics_df["experiment_cer"].mean())

print("\nSaved to:", output_file)