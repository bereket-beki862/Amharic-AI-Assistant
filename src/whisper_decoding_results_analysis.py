import pandas as pd

from config import PROJECT_ROOT

# Load decoding experiment results
input_file = PROJECT_ROOT / "data" / "processed" / "whisper_decoding_experiment.csv"

df = pd.read_csv(input_file)

print("\nWHISPER DECODING RESULTS ANALYSIS")
print("=" * 50)

print("\nDataset shape:")
print(df.shape)

print("\nAvailable columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string(index=False))

# Analyze numerical columns
numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:
    print("\nNumerical statistics:")
    print(numeric_df.describe().to_string())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum().to_string())

# Save analysis report
output_file = PROJECT_ROOT / "data" / "processed" / "whisper_decoding_results_analysis.csv"

numeric_df.describe().to_csv(output_file, encoding="utf-8-sig")

print("\nAnalysis saved to:")
print(output_file)

print("\nANALYSIS COMPLETED")