
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from config import PROCESSED_DIR

print("AMHARIC AUDIO FEATURE CLUSTERING")
print("=" * 50)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
numeric_df = numeric_df.replace(
    [float("inf"), -float("inf")], float("nan")
)

numeric_df = numeric_df.dropna(axis=1, how="all")
numeric_df = numeric_df.fillna(numeric_df.mean())

if numeric_df.shape[1] < 2:
    raise ValueError("At least two numerical features are required.")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_df)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(scaled_features)

result = pd.DataFrame({
    "sample_index": numeric_df.index,
    "cluster": clusters
})

output_file = PROCESSED_DIR / "audio_feature_clusters.csv"
result.to_csv(output_file, index=False)

plt.figure(figsize=(9, 6))
plt.scatter(
    scaled_features[:, 0],
    scaled_features[:, 1],
    c=clusters,
    alpha=0.7
)

plt.xlabel(numeric_df.columns[0])
plt.ylabel(numeric_df.columns[1])
plt.title("K-Means Clustering of Audio Features")
plt.savefig(
    PROCESSED_DIR / "audio_feature_clusters.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSamples analyzed:", len(result))
print("\nCluster distribution:")
print(result["cluster"].value_counts().sort_index())
print("\nCSV saved:", output_file)
print("Plot saved:", PROCESSED_DIR / "audio_feature_clusters.png")
print("CLUSTERING COMPLETED")
