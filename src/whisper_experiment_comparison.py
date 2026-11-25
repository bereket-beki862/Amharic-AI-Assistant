import pandas as pd
from jiwer import wer, cer

from config import PROJECT_ROOT

# Paths to experiment results
baseline_file = PROJECT_ROOT / "data" / "processed" / "whisper_evaluation.csv"
experiment_file = PROJECT_ROOT / "data" / "processed" / "whisper_decoding_experiment.csv"

# Load results
baseline_df = pd.read_csv(baseline_file)
experiment_df = pd.read_csv(experiment_file)

# Compare matching samples
comparison = []

for i in range(min(len(baseline_df), len(experiment_df))):

    reference = str(baseline_df.loc[i, "reference"])
    baseline_prediction = str(baseline_df.loc[i, "prediction"])
    experiment_prediction = str(experiment_df.loc[i, "prediction"])

    baseline_wer = wer(reference, baseline_prediction)
    experiment_wer = wer(reference, experiment_prediction)

    baseline_cer = cer(reference, baseline_prediction)
    experiment_cer = cer(reference, experiment_prediction)

    comparison.append({
        "sample": i + 1,
        "baseline_wer": baseline_wer,
        "experiment_wer": experiment_wer,
        "baseline_cer": baseline_cer,
        "experiment_cer": experiment_cer,
        "wer_improved": experiment_wer < baseline_wer,
        "cer_improved": experiment_cer < baseline_cer
    })

# Convert to DataFrame
comparison_df = pd.DataFrame(comparison)

# Calculate average metrics
avg_baseline_wer = comparison_df["baseline_wer"].mean()
avg_experiment_wer = comparison_df["experiment_wer"].mean()

avg_baseline_cer = comparison_df["baseline_cer"].mean()
avg_experiment_cer = comparison_df["experiment_cer"].mean()

# Display results
print("\nWHISPER EXPERIMENT COMPARISON")
print("=" * 45)

print(f"Baseline Average WER:   {avg_baseline_wer:.4f}")
print(f"Experiment Average WER: {avg_experiment_wer:.4f}")

print(f"\nBaseline Average CER:   {avg_baseline_cer:.4f}")
print(f"Experiment Average CER: {avg_experiment_cer:.4f}")

print("\nWER Improvement:")
print(f"{(avg_baseline_wer - avg_experiment_wer) * 100:.2f}%")

print("\nCER Improvement:")
print(f"{(avg_baseline_cer - avg_experiment_cer) * 100:.2f}%")

if avg_experiment_wer < avg_baseline_wer:
    print("\nRESULT: Experiment improved word recognition.")
elif avg_experiment_wer > avg_baseline_wer:
    print("\nRESULT: Experiment worsened word recognition.")
else:
    print("\nRESULT: No change in word recognition.")

# Save comparison report
output_file = PROJECT_ROOT / "data" / "processed" / "whisper_experiment_comparison.csv"
comparison_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nDetailed comparison saved to:")
print(output_file)