
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE CONSTANT VALUE ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    results.append({
        "feature": column,
        "unique_values": values.nunique(),
        "variance": values.var(),
        "standard_deviation": values.std(),
        "is_constant": values.nunique() <= 1
    })

result = pd.DataFrame(results)

result = result.sort_values(
    "variance",
    ascending=True
)

output_file = PROCESSED_DIR / "audio_feature_constant_report.csv"
result.to_csv(output_file, index=False)

print("\nFeature analysis:")
print(result.round(5).to_string(index=False))

print("\nConstant features:", result["is_constant"].sum())
print("Report saved:", output_file)
print("CONSTANT FEATURE ANALYSIS COMPLETED")
