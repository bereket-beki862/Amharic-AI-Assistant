
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE INTERQUARTILE RANGE ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = ((values < lower_bound) | (values > upper_bound)).sum()

    results.append({
        "feature": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outliers
    })

result = pd.DataFrame(results)

output_file = PROCESSED_DIR / "audio_feature_interquartile_report.csv"
result.to_csv(output_file, index=False)

print("\nInterquartile analysis:")
print(result.round(5).to_string(index=False))

print("\nReport saved:", output_file)
print("INTERQUARTILE ANALYSIS COMPLETED")
