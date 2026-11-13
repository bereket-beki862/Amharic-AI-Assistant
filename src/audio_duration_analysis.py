
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO DURATION ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"

df = pd.read_csv(input_file)

if "duration" not in df.columns:
    raise ValueError("The 'duration' column was not found.")

duration = df["duration"].dropna()

print("\nDuration Statistics:")
print("Total records:", len(duration))
print("Minimum:", round(duration.min(), 2), "seconds")
print("Maximum:", round(duration.max(), 2), "seconds")
print("Average:", round(duration.mean(), 2), "seconds")
print("Median:", round(duration.median(), 2), "seconds")

# Save duration statistics
output_file = PROCESSED_DIR / "audio_duration_report.csv"

duration.describe().to_csv(output_file)

# Create duration distribution chart
plt.figure(figsize=(10, 6))
plt.hist(duration, bins=20, edgecolor="black")

plt.title("Amharic Audio Duration Distribution")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Audio Files")
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_duration_distribution.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nReport saved:", output_file)
print("Chart saved:", chart_file)
print("\nAUDIO DURATION ANALYSIS COMPLETED")
