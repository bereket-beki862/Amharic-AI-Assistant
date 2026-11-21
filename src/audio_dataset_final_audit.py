
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO DATASET FINAL AUDIT")
print("=" * 60)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
categorical_df = df.select_dtypes(exclude="number")

# Basic dataset information
total_rows = len(df)
total_columns = len(df.columns)
numeric_features = len(numeric_df.columns)
categorical_features = len(categorical_df.columns)
missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

# Numerical statistics
feature_statistics = numeric_df.describe().T

# Dataset audit summary
audit = pd.DataFrame({
    "metric": [
        "Total samples",
        "Total columns",
        "Numerical features",
        "Non-numerical columns",
        "Missing values",
        "Duplicate rows",
        "Dataset has missing values",
        "Dataset has duplicate rows"
    ],
    "value": [
        total_rows,
        total_columns,
        numeric_features,
        categorical_features,
        missing_values,
        duplicate_rows,
        missing_values > 0,
        duplicate_rows > 0
    ]
})

# Save reports
audit_file = PROCESSED_DIR / "audio_dataset_final_audit.csv"
statistics_file = PROCESSED_DIR / "audio_dataset_final_statistics.csv"

audit.to_csv(audit_file, index=False)
feature_statistics.to_csv(statistics_file)

print("\nDataset Audit:")
print(audit.to_string(index=False))

print("\nNumerical Feature Statistics:")
print(feature_statistics.round(5).to_string())

print("\nAudit report saved:", audit_file)
print("Statistics saved:", statistics_file)
print("\nFINAL DATASET AUDIT COMPLETED")
