
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE SELECTION ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

# Analyze feature variance
variance = numeric_df.var()

low_variance = variance[variance < 0.01].reset_index()
low_variance.columns = ["feature", "variance"]

variance_file = PROCESSED_DIR / "audio_low_variance_features.csv"
low_variance.to_csv(variance_file, index=False)

# Analyze feature correlation
correlation_matrix = numeric_df.corr()

high_correlations = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        correlation = correlation_matrix.iloc[i, j]

        if abs(correlation) > 0.95:
            high_correlations.append({
                "feature_1": correlation_matrix.columns[i],
                "feature_2": correlation_matrix.columns[j],
                "correlation": correlation
            })

correlation_report = pd.DataFrame(
    high_correlations,
    columns=["feature_1", "feature_2", "correlation"]
)

correlation_file = PROCESSED_DIR / "audio_high_correlation_features.csv"
correlation_report.to_csv(correlation_file, index=False)

print("\nLow-variance features:")
print(low_variance.to_string(index=False))

print("\nHighly correlated feature pairs:")
print(correlation_report.round(5).to_string(index=False))

print("\nLow-variance report:", variance_file)
print("Correlation report:", correlation_file)
print("FEATURE SELECTION ANALYSIS COMPLETED")
