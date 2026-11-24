
from pathlib import Path
import pandas as pd

from config import PROCESSED_DIR


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    tsv_files = list(raw_dir.rglob("train.tsv"))

    if not tsv_files:
        print("train.tsv not found.")
        return

    df = pd.read_csv(tsv_files[0], sep="\t")

    if "sentence" not in df.columns:
        print("The 'sentence' column is missing.")
        return

    df["sentence"] = df["sentence"].fillna("").astype(str)
    df["character_count"] = df["sentence"].str.len()
    df["word_count"] = df["sentence"].str.split().str.len()

    def categorize_words(count):
        if count <= 3:
            return "Short"
        elif count <= 10:
            return "Medium"
        return "Long"

    df["length_category"] = df["word_count"].apply(categorize_words)

    print("\n--- Transcript Length Analysis ---")
    print(f"Total transcripts: {len(df)}")
    print(f"Average characters: {df['character_count'].mean():.2f}")
    print(f"Average words: {df['word_count'].mean():.2f}")

    print("\nWord count statistics:")
    print(df["word_count"].describe())

    print("\nTranscript categories:")
    print(df["length_category"].value_counts())

    output_dir = Path(PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "audio_transcript_length_analysis.csv"

    df.to_csv(output_path, index=False)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
