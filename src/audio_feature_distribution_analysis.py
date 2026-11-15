
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE DISTRIBUTION ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numeric audio features found.")

selected_features = numeric_df.var().nlargest(6).index.tolist()

print("\nSelected features:")
for feature in selected_features:
    print("-", feature)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, feature in enumerate(selected_features):
    sns.histplot(
        numeric_df[feature].dropna(),
        kde=True,
        ax=axes[i]
    )
    axes[i].set_title(feature)
    axes[i].set_xlabel("Feature Value")
    axes[i].set_ylabel("Frequency")

plt.suptitle("Amharic Audio Feature Distributions")
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_feature_distributions.png"
plt.savefig(chart_file, dpi=300)
plt.close()

report_file = PROCESSED_DIR / "audio_distribution_report.csv"
numeric_df[selected_features].describe().to_csv(report_file)

print("\nReport saved:", report_file)
print("Chart saved:", chart_file)
print("DISTRIBUTION ANALYSIS COMPLETED")
