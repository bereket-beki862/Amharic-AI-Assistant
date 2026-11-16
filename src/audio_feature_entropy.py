
import pandas as pd
import numpy as np

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE ENTROPY ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numerical features found.")

entropy_results = {}

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if len(values) == 0:
        continue

    counts, _ = np.histogram(values, bins=20)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))
    entropy_results[column] = entropy

result = pd.DataFrame(
    list(entropy_results.items()),
    columns=["feature", "entropy"]
)

result = result.sort_values(
    "entropy", ascending=False
)

output_file = PROCESSED_DIR / "audio_feature_entropy.csv"
result.to_csv(output_file, index=False)

print("\nFeature entropy results:")
print(result.round(4).to_string(index=False))

print("\nReport saved:", output_file)
print("ENTROPY ANALYSIS COMPLETED")
