
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE SPREAD ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numerical features found.")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)

    results.append({
        "feature": column,
        "minimum": values.min(),
        "maximum": values.max(),
        "range": values.max() - values.min(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "standard_deviation": values.std()
    })

result = pd.DataFrame(results)
result = result.sort_values("IQR", ascending=False)

output_file = PROCESSED_DIR / "audio_feature_spread_report.csv"

if output_file.exists():
    raise FileExistsError(
        f"Report already exists: {output_file}. "
        "No existing file was overwritten."
    )

result.to_csv(output_file, index=False)

print("\nFeature spread results:")
print(result.round(4).to_string(index=False))

print("\nReport saved:", output_file)
print("SPREAD ANALYSIS COMPLETED")
