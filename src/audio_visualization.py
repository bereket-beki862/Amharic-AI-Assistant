import pandas as pd
import matplotlib.pyplot as plt
from config import PROCESSED_DIR

df = pd.read_csv(PROCESSED_DIR / "audio_features.csv")

numeric_df = df.select_dtypes(include="number")

numeric_df.hist(figsize=(14, 10), bins=20)
plt.suptitle("Amharic Audio Feature Distributions")
plt.tight_layout()

output = PROCESSED_DIR / "audio_feature_distributions.png"
plt.savefig(output, dpi=300)
plt.close()

print("Visualization saved:", output)
print("AUDIO VISUALIZATION COMPLETED")