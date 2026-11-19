
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE COEFFICIENT OF VARIATION ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    values = numeric_df[column].dropna()

    if len(values) < 2:
        continue

    mean_value = values.mean()
    std_value = values.std()

    cv = (
        (std_value / abs(mean_value)) * 100
        if mean_value != 0
        else None
    )

    results.append({
        "feature": column,
        "mean": mean_value,
        "standard_deviation": std_value,
        "coefficient_of_variation_percent": cv
    })

result = pd.DataFrame(results)

result = result.sort_values(
    "coefficient_of_variation_percent",
    ascending=False,
    na_position="last"
)

output_file = PROCESSED_DIR / "audio_feature_coefficient_variation.csv"
result.to_csv(output_file, index=False)

print("\nCoefficient of variation report:")
print(result.round(5).to_string(index=False))

print("\nReport saved:", output_file)
print("COEFFICIENT OF VARIATION ANALYSIS COMPLETED")
