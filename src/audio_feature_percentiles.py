
import pandas as pd

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE PERCENTILE ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numerical features found.")

percentiles = numeric_df.describe(
    percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
).T

output_file = PROCESSED_DIR / "audio_feature_percentiles.csv"
percentiles.to_csv(output_file)

print("\nFeature percentile results:")
print(percentiles.round(3).to_string())

print("\nReport saved:", output_file)
print("PERCENTILE ANALYSIS COMPLETED")
S