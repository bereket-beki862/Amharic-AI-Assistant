
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from config import PROCESSED_DIR

print("AUDIO FEATURE SCALING COMPARISON")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

numeric_df = df.select_dtypes(include="number")

if numeric_df.empty:
    print("No numerical features found.")
else:
    scaling_methods = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }

    for name, scaler in scaling_methods.items():
        scaled_data = scaler.fit_transform(numeric_df)

        scaled_df = pd.DataFrame(
            scaled_data,
            columns=numeric_df.columns,
            index=numeric_df.index
        )

        output_file = PROCESSED_DIR / f"audio_features_{name}_scaled.csv"
        scaled_df.to_csv(output_file, index=False)

        print(f"\n{name.capitalize()} scaling:")
        print("Saved:", output_file)
        print("Mean:", scaled_df.mean().mean())
        print("Standard deviation:", scaled_df.std().mean())

print("\nSCALING COMPARISON COMPLETED")
