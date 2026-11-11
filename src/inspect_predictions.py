
import pandas as pd
from config import PROJECT_ROOT

file = PROJECT_ROOT / "data" / "processed" / "whisper_evaluation.csv"

df = pd.read_csv(file)

for i, row in df.iterrows():
    print("\n" + "=" * 60)
    print("SAMPLE:", i + 1)
    print("REFERENCE:")
    print(row["reference"])
    print("PREDICTION:")
    print(row["prediction"])

print("\nINSPECTION COMPLETED")
