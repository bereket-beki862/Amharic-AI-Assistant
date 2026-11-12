
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE CORRELATION ANALYSIS")
print("=" * 45)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

output_file = PROCESSED_DIR / "feature_correlation.csv"
correlation.to_csv(output_file)

pairs = correlation.abs().where(
    ~pd.DataFrame(
        [[i == j for j in correlation.columns]
         for i in correlation.columns],
        index=correlation.index,
        columns=correlation.columns
    )
).stack().sort_values(ascending=False)

print("Records:", len(df))
print("Numeric features:", len(numeric_df.columns))
print("\nTop 5 correlated feature pairs:")
print(pairs.head(5))

print("\nSaved:", output_file)
print("\nCORRELATION ANALYSIS COMPLETED")
