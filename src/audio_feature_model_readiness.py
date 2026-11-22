
import pandas as pd
import numpy as np
from config import PROCESSED_DIR

print("AUDIO FEATURE MODEL READINESS CHECK")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

missing_values = numeric_df.isna().sum().sum()
infinite_values = np.isinf(numeric_df).sum().sum()
constant_features = [
    column for column in numeric_df.columns
    if numeric_df[column].nunique(dropna=True) <= 1
]
duplicate_rows = df.duplicated().sum()

checks = [
    ("Dataset loaded", True),
    ("Numerical features available", len(numeric_df.columns) > 0),
    ("Missing numerical values", missing_values == 0),
    ("Infinite values", infinite_values == 0),
    ("Constant features", len(constant_features) == 0),
    ("Duplicate rows", duplicate_rows == 0)
]

report = pd.DataFrame(checks, columns=["check", "passed"])

print("\nDataset information:")
print("Samples:", len(df))
print("Total columns:", len(df.columns))
print("Numerical features:", len(numeric_df.columns))

print("\nReadiness checks:")
for check, passed in checks:
    status = "PASS" if passed else "WARNING"
    print(f"{status}: {check}")

print("\nIssues found:")
print("Missing numerical values:", missing_values)
print("Infinite values:", infinite_values)
print("Constant features:", constant_features)
print("Duplicate rows:", duplicate_rows)

ready_count = report["passed"].sum()
total_checks = len(report)

print(f"\nReadiness score: {ready_count}/{total_checks}")

output_file = PROCESSED_DIR / "audio_feature_model_readiness_report.csv"
report.to_csv(output_file, index=False)

print("\nReport saved:", output_file)
print("MODEL READINESS CHECK COMPLETED")
