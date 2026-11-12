import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE SUMMARY")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

print("\nDataset shape:", df.shape)
print("\nFeature statistics:")
print(numeric_df.describe().round(3))

summary_file = PROCESSED_DIR / "audio_feature_summary.csv"
numeric_df.describe().T.to_csv(summary_file)

print("\nSummary saved:", summary_file)
print("\nFEATURE SUMMARY COMPLETED")