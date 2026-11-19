
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE Z-SCORE OUTLIER ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column]

    mean_value = values.mean()
    std_value = values.std()

    if pd.isna(std_value) or std_value == 0:
        continue

    z_scores = (values - mean_value) / std_value
    outlier_count = (z_scores.abs() > 3).sum()

    results.append({
        "feature": column,
        "mean": mean_value,
        "standard_deviation": std_value,
        "outlier_count": outlier_count,
        "outlier_percentage": (outlier_count / values.notna().sum()) * 100
    })

result = pd.DataFrame(results)

result = result.sort_values(
    "outlier_count",
    ascending=False
)

output_file = PROCESSED_DIR / "audio_feature_zscore_outliers.csv"
result.to_csv(output_file, index=False)

print("\nZ-score outlier report:")
print(result.round(5).to_string(index=False))

print("\nReport saved:", output_file)
print("Z-SCORE OUTLIER ANALYSIS COMPLETED")
