
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE MAD OUTLIER ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    median_value = values.median()
    mad = (values - median_value).abs().median()

    if mad == 0:
        outlier_count = 0
    else:
        modified_z_scores = (
            0.6745 * (values - median_value) / mad
        )
        outlier_count = (modified_z_scores.abs() > 3.5).sum()

    results.append({
        "feature": column,
        "median": median_value,
        "MAD": mad,
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_count / len(values) * 100
    })

result = pd.DataFrame(results)
result = result.sort_values("outlier_count", ascending=False)

output_file = PROCESSED_DIR / "audio_feature_mad_outliers.csv"
result.to_csv(output_file, index=False)

print("\nMAD outlier report:")
print(result.round(5).to_string(index=False))

print("\nReport saved:", output_file)
print("MAD OUTLIER ANALYSIS COMPLETED")
