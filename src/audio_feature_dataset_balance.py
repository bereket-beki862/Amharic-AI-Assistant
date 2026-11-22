
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE DATASET BALANCE ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

possible_labels = [
    "label", "target", "class", "category",
    "gender", "speaker", "accent"
]

label_column = next(
    (column for column in df.columns
     if column.lower() in possible_labels),
    None
)

if label_column is None:
    print("\nNo recognized label column found.")
    print("Dataset columns:")
    print(df.columns.tolist())
else:
    print("\nLabel column:", label_column)

    counts = df[label_column].value_counts(dropna=False)
    percentages = df[label_column].value_counts(
        dropna=False, normalize=True
    ) * 100

    balance_report = pd.DataFrame({
        "class": counts.index.astype(str),
        "sample_count": counts.values,
        "percentage": percentages.values
    })

    balance_report["percentage"] = balance_report["percentage"].round(2)

    output_file = PROCESSED_DIR / "audio_feature_dataset_balance.csv"
    balance_report.to_csv(output_file, index=False)

    print("\nClass distribution:")
    print(balance_report.to_string(index=False))

    if len(counts) > 1:
        largest_class = counts.max()
        smallest_class = counts.min()

        imbalance_ratio = (
            largest_class / smallest_class
            if smallest_class > 0
            else float("inf")
        )

        print("\nNumber of classes:", len(counts))
        print("Largest class:", largest_class)
        print("Smallest class:", smallest_class)
        print("Imbalance ratio:", round(imbalance_ratio, 4))

    print("\nReport saved:", output_file)

print("\nDATASET BALANCE ANALYSIS COMPLETED")
