
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE CORRELATION STRENGTH ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
correlation_matrix = numeric_df.corr()

results = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        feature_1 = correlation_matrix.columns[i]
        feature_2 = correlation_matrix.columns[j]
        correlation = correlation_matrix.iloc[i, j]
        strength = abs(correlation)

        if strength >= 0.7:
            category = "Strong"
        elif strength >= 0.4:
            category = "Moderate"
        else:
            category = "Weak"

        results.append({
            "feature_1": feature_1,
            "feature_2": feature_2,
            "correlation": correlation,
            "strength": category
        })

result = pd.DataFrame(results)

result = result.sort_values(
    "correlation",
    key=lambda values: values.abs(),
    ascending=False
)

output_file = PROCESSED_DIR / "audio_feature_correlation_strength.csv"
result.to_csv(output_file, index=False)

print("\nCorrelation strength report:")
print(result.round(5).to_string(index=False))

print("\nStrong relationships:",
      (result["strength"] == "Strong").sum())
print("Report saved:", output_file)
print("CORRELATION STRENGTH ANALYSIS COMPLETED")
