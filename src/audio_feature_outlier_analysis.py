
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE OUTLIER ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

results = []

for column in numeric_df.columns:
    q1 = numeric_df[column].quantile(0.25)
    q3 = numeric_df[column].quantile(0.75)

    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = (
        (numeric_df[column] < lower_limit) |
        (numeric_df[column] > upper_limit)
    )

    results.append({
        "Feature": column,
        "Outlier_Count": int(outliers.sum()),
        "Outlier_Percentage": round(
            outliers.mean() * 100, 2
        )
    })

report = pd.DataFrame(results)
report = report.sort_values(
    by="Outlier_Count",
    ascending=False
)

output_file = PROCESSED_DIR / "audio_outlier_report.csv"
report.to_csv(output_file, index=False)

print("\nTop 10 features by outlier count:")
print(report.head(10).to_string(index=False))

plt.figure(figsize=(12, 7))

top_features = report.head(10)

plt.barh(
    top_features["Feature"],
    top_features["Outlier_Count"]
)

plt.xlabel("Number of Outliers")
plt.ylabel("Audio Features")
plt.title("Top 10 Audio Features by Outlier Count")
plt.gca().invert_yaxis()
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_outlier_distribution.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nReport saved:", output_file)
print("Chart saved:", chart_file)
print("\nOUTLIER ANALYSIS COMPLETED")
