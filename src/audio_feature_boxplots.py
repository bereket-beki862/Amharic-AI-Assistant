
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE BOXPLOT ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    raise ValueError("No numeric audio features found.")

# Select up to 12 numeric features
selected_features = numeric_df.columns[:12]

plt.figure(figsize=(14, 7))
numeric_df[selected_features].boxplot(rot=45)

plt.title("Amharic Audio Feature Distributions")
plt.ylabel("Feature Values")
plt.tight_layout()

output_file = PROCESSED_DIR / "audio_feature_boxplots.png"
plt.savefig(output_file, dpi=300)
plt.close()

print("Features analyzed:", len(selected_features))
print("Boxplot saved:", output_file)
print("BOXPLOT ANALYSIS COMPLETED")
