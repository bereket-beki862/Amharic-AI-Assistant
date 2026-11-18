
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE RANGE ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    minimum = values.min()
    maximum = values.max()

    results.append({
        "feature": column,
        "minimum": minimum,
        "maximum": maximum,
        "range": maximum - minimum,
        "mean": values.mean()
    })

result = pd.DataFrame(results)
result = result.sort_values("range", ascending=False)

output_file = PROCESSED_DIR / "audio_feature_range_report.csv"
result.to_csv(output_file, index=False)

print("\nFeature range report:")
print(result.round(5).to_string(index=False))

print("\nReport saved:", output_file)
print("RANGE ANALYSIS COMPLETED")
