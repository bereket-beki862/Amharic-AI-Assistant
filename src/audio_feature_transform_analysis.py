
import pandas as pd
import numpy as np
from config import PROCESSED_DIR

print("AUDIO FEATURE TRANSFORMATION ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []
transformed_columns = {}

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if values.empty:
        continue

    original_skewness = values.skew()

    if values.min() >= 0:
        transformed = np.log1p(values)
        method = "log1p"
    else:
        transformed = np.sign(values) * np.log1p(np.abs(values))
        method = "signed_log1p"

    transformed_skewness = pd.Series(transformed).skew()

    results.append({
        "feature": column,
        "transformation": method,
        "original_skewness": original_skewness,
        "transformed_skewness": transformed_skewness,
        "skewness_change": abs(original_skewness) - abs(transformed_skewness)
    })

    transformed_columns[column] = transformed

result = pd.DataFrame(results)
result = result.sort_values("skewness_change", ascending=False)

report_file = PROCESSED_DIR / "audio_feature_transform_report.csv"
result.to_csv(report_file, index=False)

transformed_df = pd.DataFrame(transformed_columns)
transformed_file = PROCESSED_DIR / "audio_features_log_transformed.csv"
transformed_df.to_csv(transformed_file, index=False)

print("\nTransformation report:")
print(result.round(5).to_string(index=False))

print("\nFeatures with reduced absolute skewness:",
      (result["skewness_change"] > 0).sum())

print("\nReport saved:", report_file)
print("Transformed features saved:", transformed_file)
print("TRANSFORMATION ANALYSIS COMPLETED")
