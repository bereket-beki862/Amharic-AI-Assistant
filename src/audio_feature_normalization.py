
import pandas as pd

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE NORMALIZATION")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numeric features found.")

# Calculate mean and standard deviation
means = numeric_df.mean()
stds = numeric_df.std().replace(0, 1)

# Apply Z-score normalization
normalized_df = (numeric_df - means) / stds

# Save normalized features
output_file = PROCESSED_DIR / "normalized_audio_features.csv"
normalized_df.to_csv(output_file, index=False)

print("\nOriginal dataset shape:", numeric_df.shape)
print("Normalized dataset shape:", normalized_df.shape)

print("\nNormalized feature statistics:")
print(normalized_df.describe().round(3))

print("\nNormalized dataset saved:", output_file)
print("NORMALIZATION COMPLETED")
