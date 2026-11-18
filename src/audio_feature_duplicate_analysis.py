import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE DUPLICATE ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

duplicate_rows = df.duplicated()
duplicate_count = duplicate_rows.sum()

print("\nDataset shape:", df.shape)
print("Duplicate rows found:", duplicate_count)

duplicates = df[duplicate_rows]

output_file = PROCESSED_DIR / "audio_feature_duplicates.csv"
duplicates.to_csv(output_file, index=False)

print("\nDuplicate records saved:", output_file)
print("DUPLICATE ANALYSIS COMPLETED")