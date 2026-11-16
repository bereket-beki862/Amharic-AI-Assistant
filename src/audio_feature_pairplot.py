
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE PAIRPLOT ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

selected_features = [
    "duration",
    "spectral_centroid",
    "zero_crossing_rate",
    "rms"
]

available_features = [
    feature for feature in selected_features
    if feature in df.columns
]

if len(available_features) < 2:
    raise ValueError("Not enough selected features found.")

plot_df = df[available_features].dropna()

scatter_matrix(
    plot_df,
    figsize=(12, 10),
    diagonal="hist",
    alpha=0.6
)

plt.suptitle("Audio Feature Pairplot", y=1.02)
plt.tight_layout()

output_file = PROCESSED_DIR / "audio_feature_pairplot.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print("\nSelected features:", available_features)
print("Samples analyzed:", len(plot_df))
print("Plot saved:", output_file)
print("PAIRPLOT ANALYSIS COMPLETED")
