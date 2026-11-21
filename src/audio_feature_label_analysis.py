
import pandas as pd
from config import PROCESSED_DIR

print("AUDIO FEATURE LABEL ANALYSIS")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

# Identify likely label columns
possible_labels = [
    column for column in df.columns
    if column.lower() in [
        "label", "target", "class", "category",
        "gender", "speaker", "accent"
    ]
]

if not possible_labels:
    print("No recognized label column found.")
    print("Available columns:", df.columns.tolist())
else:
    label_column = possible_labels[0]

    print("\nLabel column detected:", label_column)
    print("Label distribution:")
    print(df[label_column].value_counts(dropna=False).to_string())

    numeric_df = df.select_dtypes(include="number")

    # Feature means grouped by label
    grouped_means = numeric_df.groupby(df[label_column]).mean()

    output_file = PROCESSED_DIR / "audio_feature_label_means.csv"
    grouped_means.to_csv(output_file)

    print("\nMean feature values per label:")
    print(grouped_means.round(5).to_string())

    # Compare feature means across labels
    if len(grouped_means) >= 2:
        difference = grouped_means.max() - grouped_means.min()

        comparison = pd.DataFrame({
            "feature": difference.index,
            "mean_difference_between_labels": difference.values
        })

        comparison = comparison.sort_values(
            "mean_difference_between_labels",
            ascending=False
        )

        comparison_file = PROCESSED_DIR / "audio_feature_label_comparison.csv"
        comparison.to_csv(comparison_file, index=False)

        print("\nLargest feature differences between labels:")
        print(comparison.round(5).to_string(index=False))
        print("Comparison saved:", comparison_file)

    print("\nGrouped means saved:", output_file)

print("\nLABEL ANALYSIS COMPLETED")
