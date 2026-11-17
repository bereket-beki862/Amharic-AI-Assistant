import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE ENTROPY COMPARISON")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_feature_entropy.csv"
df = pd.read_csv(input_file)

if not {"feature", "entropy"}.issubset(df.columns):
    raise ValueError("Unexpected entropy report format.")

df = df.sort_values("entropy", ascending=False)

print("\nHighest entropy features:")
print(df.head(10).to_string(index=False))

print("\nLowest entropy features:")
print(df.tail(10).sort_values("entropy").to_string(index=False))

print("\nAverage entropy:", round(df["entropy"].mean(), 4))
print("Median entropy:", round(df["entropy"].median(), 4))

print("\nENTROPY COMPARISON COMPLETED")