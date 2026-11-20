
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE DISTRIBUTION SUMMARY")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

summary = pd.DataFrame({
    "mean": numeric_df.mean(),
    "median": numeric_df.median(),
    "minimum": numeric_df.min(),
    "maximum": numeric_df.max(),
    "standard_deviation": numeric_df.std(),
    "variance": numeric_df.var(),
    "skewness": numeric_df.skew(),
    "kurtosis": numeric_df.kurt(),
    "Q1": numeric_df.quantile(0.25),
    "Q3": numeric_df.quantile(0.75)
})

summary["IQR"] = summary["Q3"] - summary["Q1"]

summary.index.name = "feature"

output_file = PROCESSED_DIR / "audio_feature_distribution_summary.csv"
summary.to_csv(output_file)

print("\nDistribution summary:")
print(summary.round(5).to_string())

print("\nTotal numerical features:", len(numeric_df.columns))
print("Report saved:", output_file)
print("DISTRIBUTION SUMMARY COMPLETED")
