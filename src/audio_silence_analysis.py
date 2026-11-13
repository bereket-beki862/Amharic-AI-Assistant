
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO SILENCE ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

if "zero_crossing_rate" not in df.columns:
    raise ValueError("zero_crossing_rate column not found.")

zcr = df["zero_crossing_rate"].dropna()

# Analyze zero-crossing rate
print("\nZero-Crossing Rate Statistics:")
print("Total records:", len(zcr))
print("Minimum:", round(zcr.min(), 4))
print("Maximum:", round(zcr.max(), 4))
print("Average:", round(zcr.mean(), 4))
print("Median:", round(zcr.median(), 4))

# Save report
report_file = PROCESSED_DIR / "audio_silence_report.csv"
zcr.describe().to_csv(report_file)

# Create visualization
plt.figure(figsize=(10, 6))
plt.hist(zcr, bins=25, edgecolor="black")

plt.title("Amharic Audio Zero-Crossing Rate Distribution")
plt.xlabel("Zero-Crossing Rate")
plt.ylabel("Number of Audio Files")
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_silence_distribution.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nReport saved:", report_file)
print("Chart saved:", chart_file)
print("\nAUDIO SILENCE ANALYSIS COMPLETED")
