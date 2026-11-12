import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AUDIO FEATURE CORRELATION HEATMAP")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False,
    linewidths=0.3
)

plt.title("Amharic Audio Feature Correlation")
plt.tight_layout()

output_file = PROCESSED_DIR / "audio_correlation_heatmap.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("Heatmap saved:", output_file)
print("HEATMAP GENERATION COMPLETED")