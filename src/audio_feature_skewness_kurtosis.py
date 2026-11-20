
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE SKEWNESS AND KURTOSIS ANALYSIS")
print("=" * 60)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if len(values) < 4:
        continue

    skewness = values.skew()
    kurtosis = values.kurt()

    results.append({
        "feature": column,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "distribution_shape": (
            "Right-skewed" if skewness > 0.5
            else "Left-skewed" if skewness < -0.5
            else "Approximately symmetrical"
        ),
        "kurtosis_type": (
            "Heavy-tailed" if kurtosis > 0
            else "Light-tailed" if kurtosis < 0
            else "Normal-like tails"
        )
    })

result = pd.DataFrame(results)

output_file = PROCESSED_DIR / "audio_feature_skewness_kurtosis_report.csv"
result.to_csv(output_file, index=False)

print("\nSkewness and kurtosis report:")
print(result.round(5).to_string(index=False))

print("\nReport saved:", output_file)
print("SKEWNESS AND KURTOSIS ANALYSIS COMPLETED")
