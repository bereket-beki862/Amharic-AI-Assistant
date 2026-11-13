
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO SPECTRAL ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

features = [
    col for col in [
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff"
    ]
    if col in df.columns
]

if not features:
    raise ValueError("No spectral features found.")

spectral_data = df[features]

print("\nSpectral Feature Statistics:")
print(spectral_data.describe().round(3))

report_file = PROCESSED_DIR / "audio_spectral_report.csv"
spectral_data.describe().to_csv(report_file)

spectral_data.hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Amharic Audio Spectral Feature Distributions")
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_spectral_distribution.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nReport saved:", report_file)
print("Chart saved:", chart_file)
print("\nSPECTRAL ANALYSIS COMPLETED")
