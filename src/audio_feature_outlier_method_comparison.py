
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE OUTLIER METHOD COMPARISON")
print("=" * 60)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    # IQR method
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    iqr_count = (
        (values < q1 - 1.5 * iqr) |
        (values > q3 + 1.5 * iqr)
    ).sum()

    # Z-score method
    std_value = values.std()

    if pd.isna(std_value) or std_value == 0:
        zscore_count = 0
    else:
        z_scores = (values - values.mean()) / std_value
        zscore_count = (z_scores.abs() > 3).sum()

    # MAD method
    median_value = values.median()
    mad = (values - median_value).abs().median()

    if mad == 0:
        mad_count = 0
    else:
        modified_z = 0.6745 * (values - median_value) / mad
        mad_count = (modified_z.abs() > 3.5).sum()

    results.append({
        "feature": column,
        "IQR_outliers": iqr_count,
        "Zscore_outliers": zscore_count,
        "MAD_outliers": mad_count
    })

result = pd.DataFrame(results)

output_file = PROCESSED_DIR / "audio_feature_outlier_comparison.csv"
result.to_csv(output_file, index=False)

print("\nOutlier comparison report:")
print(result.to_string(index=False))

print("\nTotal IQR outliers:", result["IQR_outliers"].sum())
print("Total Z-score outliers:", result["Zscore_outliers"].sum())
print("Total MAD outliers:", result["MAD_outliers"].sum())

print("\nReport saved:", output_file)
print("OUTLIER METHOD COMPARISON COMPLETED")
