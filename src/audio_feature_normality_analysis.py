
import pandas as pd
from scipy.stats import shapiro
from config import PROCESSED_DIR

print("AUDIO FEATURE NORMALITY ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if len(values) < 3 or values.nunique() <= 1:
        continue

    statistic, p_value = shapiro(values)

    results.append({
        "feature": column,
        "sample_count": len(values),
        "shapiro_statistic": statistic,
        "p_value": p_value,
        "is_approximately_normal": p_value > 0.05
    })

result = pd.DataFrame(results)

output_file = PROCESSED_DIR / "audio_feature_normality_report.csv"
result.to_csv(output_file, index=False)

print("\nNormality test results:")
print(result.round(5).to_string(index=False))

print("\nFeatures approximately normal:",
      result["is_approximately_normal"].sum())

print("Report saved:", output_file)
print("NORMALITY ANALYSIS COMPLETED")
