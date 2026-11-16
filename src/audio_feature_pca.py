
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE PCA ANALYSIS")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
numeric_df = numeric_df.replace([float("inf"), -float("inf")], float("nan"))
numeric_df = numeric_df.dropna(axis=1, how="all")
numeric_df = numeric_df.fillna(numeric_df.mean())

if numeric_df.shape[1] < 2:
    raise ValueError("At least two numerical features are required.")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_df)

pca = PCA(n_components=2)
components = pca.fit_transform(scaled_features)

pca_df = pd.DataFrame(
    components,
    columns=["PC1", "PC2"]
)

output_file = PROCESSED_DIR / "audio_feature_pca.csv"
pca_df.to_csv(output_file, index=False)

plt.figure(figsize=(10, 7))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Audio Features - PCA Visualization")
plt.grid(alpha=0.3)

plot_file = PROCESSED_DIR / "audio_feature_pca.png"
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()

print("\nSamples analyzed:", len(pca_df))
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", round(sum(pca.explained_variance_ratio_), 4))
print("CSV saved:", output_file)
print("Plot saved:", plot_file)
print("PCA ANALYSIS COMPLETED")
