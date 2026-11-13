
import pandas as pd
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

print("AMHARIC AUDIO MFCC ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

mfcc_columns = [
    col for col in df.columns
    if col.startswith("mfcc_")
]

if not mfcc_columns:
    raise ValueError("No MFCC features found in the dataset.")

mfcc_data = df[mfcc_columns]

print("\nMFCC Analysis:")
print("Total records:", len(mfcc_data))
print("MFCC features:", len(mfcc_columns))

print("\nAverage MFCC values:")
print(mfcc_data.mean().round(4))

# Save MFCC statistics
report_file = PROCESSED_DIR / "audio_mfcc_report.csv"
mfcc_data.describe().to_csv(report_file)

# Plot average MFCC values
plt.figure(figsize=(12, 6))
mfcc_data.mean().plot(kind="bar", edgecolor="black")

plt.title("Average MFCC Features in Amharic Audio")
plt.xlabel("MFCC Coefficients")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.tight_layout()

chart_file = PROCESSED_DIR / "audio_mfcc_distribution.png"
plt.savefig(chart_file, dpi=300)
plt.close()

print("\nReport saved:", report_file)
print("Chart saved:", chart_file)
print("\nMFCC ANALYSIS COMPLETED")
