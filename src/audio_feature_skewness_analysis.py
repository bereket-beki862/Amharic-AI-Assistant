
import pandas as pd

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE SKEWNESS ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numeric features found.")

results = pd.DataFrame({
    "Feature": numeric_df.columns,
    "Mean": numeric_df.mean().values,
    "Skewness": numeric_df.skew().values,
    "Kurtosis": numeric_df.kurtosis().values
})

results = results.round(4)
results["Absolute_Skewness"] = results["Skewness"].abs()
results = results.sort_values(
    by="Absolute_Skewness",
    ascending=False
)

output_file = PROCESSED_DIR / "audio_skewness_report.csv"
results.to_csv(output_file, index=False)

print("\nTop 10 features by absolute skewness:")
print(results.head(10).to_string(index=False))

print("\nReport saved:", output_file)
print("SKEWNESS ANALYSIS COMPLETED")
