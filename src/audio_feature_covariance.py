
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE COVARIANCE ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numeric features found.")

# Calculate covariance matrix
covariance = numeric_df.cov()

# Save covariance report
report_file = PROCESSED_DIR / "audio_covariance_report.csv"
covariance.to_csv(report_file)

# Generate covariance heatmap
plt.figure(figsize=(16, 12))

sns.heatmap(
    covariance,
    cmap="coolwarm",
    center=0,
    linewidths=0.3
)

plt.title("Amharic Audio Feature Covariance Matrix")
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_covariance_heatmap.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nCovariance matrix shape:", covariance.shape)
print("Report saved:", report_file)
print("Heatmap saved:", chart_file)
print("COVARIANCE ANALYSIS COMPLETED")
