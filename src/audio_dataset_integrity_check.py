
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO DATASET INTEGRITY CHECK")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

print("\nDataset dimensions")
print("-" * 30)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Missing values
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

# Duplicate rows
duplicate_rows = df.duplicated().sum()

# Empty columns
empty_columns = df.columns[df.isnull().all()].tolist()

# Duplicate column names
duplicate_column_names = df.columns[df.columns.duplicated()].tolist()

# Integrity summary
integrity_report = pd.DataFrame({
    "check": [
        "Total rows",
        "Total columns",
        "Total missing values",
        "Duplicate rows",
        "Completely empty columns",
        "Duplicate column names"
    ],
    "result": [
        df.shape[0],
        df.shape[1],
        total_missing,
        duplicate_rows,
        len(empty_columns),
        len(duplicate_column_names)
    ]
})

print("\nIntegrity Summary")
print("-" * 30)
print(integrity_report.to_string(index=False))

print("\nMissing values by column")
print("-" * 30)
print(missing_values[missing_values > 0].to_string())

print("\nEmpty columns:", empty_columns)
print("Duplicate column names:", duplicate_column_names)

output_file = PROCESSED_DIR / "audio_dataset_integrity_report.csv"
integrity_report.to_csv(output_file, index=False)

print("\nReport saved:", output_file)
print("DATASET INTEGRITY CHECK COMPLETED")
