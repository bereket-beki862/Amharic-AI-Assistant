
import pandas as pd
from config import PROCESSED_DIR

print("AMHARIC AUDIO QUALITY ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Dataset not found: {input_file}")

df = pd.read_csv(input_file)

print("Total recordings:", len(df))
print("Total feature columns:", len(df.columns))

print("\nMissing values:")
print(df.isnull().sum().sum())

numeric_df = df.select_dtypes(include="number")

print("\nFeature statistics:")
print(numeric_df.describe().round(3).to_string())

print("\nConstant features:")
constant_features = [
    col for col in numeric_df.columns
    if numeric_df[col].nunique() <= 1
]

if constant_features:
    print(constant_features)
else:
    print("None")

print("\nDuplicate rows:", df.duplicated().sum())

output_file = PROCESSED_DIR / "audio_quality_report.csv"

numeric_df.describe().T.to_csv(output_file)

print("\nReport saved to:", output_file)
print("\nAUDIO QUALITY ANALYSIS COMPLETED")
