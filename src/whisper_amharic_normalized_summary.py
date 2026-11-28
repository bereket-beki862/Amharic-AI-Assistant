import pandas as pd
from config import PROJECT_ROOT

# Load normalized metrics
input_file = (
    PROJECT_ROOT / "data" / "processed"
    / "whisper_amharic_normalized_metrics.csv"
)

df = pd.read_csv(input_file)

print("=" * 50)
print("WHISPER AMHARIC NORMALIZED SUMMARY")
print("=" * 50)

if df.empty:
    print("No evaluation data found.")
else:
    # Calculate overall metrics
    avg_wer = df["normalized_wer"].mean()
    avg_cer = df["normalized_cer"].mean()

    best_wer = df.loc[df["normalized_wer"].idxmin()]
    worst_wer = df.loc[df["normalized_wer"].idxmax()]

    best_cer = df.loc[df["normalized_cer"].idxmin()]
    worst_cer = df.loc[df["normalized_cer"].idxmax()]

    print("\nTotal samples:", len(df))

    print(f"\nAverage WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")

    print("\nBEST WER SAMPLE")
    print("Sample:", int(best_wer["sample"]))
    print("WER:", best_wer["normalized_wer"])

    print("\nWORST WER SAMPLE")
    print("Sample:", int(worst_wer["sample"]))
    print("WER:", worst_wer["normalized_wer"])

    print("\nBEST CER SAMPLE")
    print("Sample:", int(best_cer["sample"]))
    print("CER:", best_cer["normalized_cer"])

    print("\nWORST CER SAMPLE")
    print("Sample:", int(worst_cer["sample"]))
    print("CER:", worst_cer["normalized_cer"])

    # Save summary
    summary = pd.DataFrame({
        "metric": [
            "total_samples",
            "average_normalized_wer",
            "average_normalized_cer",
            "best_wer_sample",
            "worst_wer_sample",
            "best_cer_sample",
            "worst_cer_sample"
        ],
        "value": [
            len(df),
            avg_wer,
            avg_cer,
            int(best_wer["sample"]),
            int(worst_wer["sample"]),
            int(best_cer["sample"]),
            int(worst_cer["sample"])
        ]
    })

    output_file = (
        PROJECT_ROOT / "data" / "processed"
        / "whisper_amharic_normalized_summary.csv"
    )

    summary.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nSummary saved to:")
    print(output_file)

print("\nSUMMARY COMPLETED")