
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE CORRELATION NETWORK ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
correlation_matrix = numeric_df.corr()

threshold = 0.80
relationships = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        feature_1 = correlation_matrix.columns[i]
        feature_2 = correlation_matrix.columns[j]
        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) >= threshold:
            relationships.append({
                "feature_1": feature_1,
                "feature_2": feature_2,
                "correlation": correlation,
                "relationship": (
                    "Positive" if correlation > 0 else "Negative"
                )
            })

result = pd.DataFrame(
    relationships,
    columns=["feature_1", "feature_2", "correlation", "relationship"]
)

result = result.sort_values(
    "correlation",
    key=lambda values: values.abs(),
    ascending=False
)

output_file = PROCESSED_DIR / "audio_feature_correlation_network.csv"
result.to_csv(output_file, index=False)

print("\nStrong feature relationships (|correlation| >= 0.80):")
print(result.round(5).to_string(index=False))

print("\nStrong relationships found:", len(result))
print("Report saved:", output_file)
print("CORRELATION NETWORK ANALYSIS COMPLETED")
