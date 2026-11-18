import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE MISSING VALUES ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

result = pd.DataFrame({
    "feature": df.columns,
    "missing_values": missing_count.values,
    "missing_percentage": missing_percentage.values
})

result = result.sort_values(
    "missing_values",
    ascending=False
)

output_file = PROCESSED_DIR / "audio_feature_missing_values.csv"
result.to_csv(output_file, index=False)

print("\nMissing value report:")
print(result.to_string(index=False))

print("\nTotal missing values:", missing_count.sum())
print("Report saved:", output_file)
print("MISSING VALUE ANALYSIS COMPLETED")