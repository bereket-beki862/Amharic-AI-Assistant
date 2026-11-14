
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE VARIANCE ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numeric features found.")

variance = numeric_df.var().sort_values(ascending=False)

report = pd.DataFrame({
    "Feature": variance.index,
    "Variance": variance.values
})

report["Variance"] = report["Variance"].round(6)

output_file = PROCESSED_DIR / "audio_variance_report.csv"
report.to_csv(output_file, index=False)

print("\nTop 10 features by variance:")
print(report.head(10).to_string(index=False))

top_features = variance.head(10)

plt.figure(figsize=(12, 7))
plt.barh(top_features.index, top_features.values)

plt.xlabel("Variance")
plt.ylabel("Audio Features")
plt.title("Top 10 Audio Features by Variance")
plt.gca().invert_yaxis()
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_variance_distribution.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nReport saved:", output_file)
print("Chart saved:", chart_file)
print("VARIANCE ANALYSIS COMPLETED")
