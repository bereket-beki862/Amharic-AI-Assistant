
from pathlib import Path
import pandas as pd

from config import PROCESSED_DIR


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = " ".join(text.split())
    return text.strip()


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    tsv_files = list(raw_dir.rglob("train.tsv"))

    if not tsv_files:
        print("train.tsv not found.")
        return

    input_path = tsv_files[0]
    df = pd.read_csv(input_path, sep="\t")

    if "sentence" not in df.columns:
        print("The sentence column was not found.")
        return

    original_count = len(df)

    df["sentence_original"] = df["sentence"]
    df["sentence_cleaned"] = df["sentence"].apply(clean_text)

    empty_count = (df["sentence_cleaned"] == "").sum()

    cleaned_df = df[df["sentence_cleaned"] != ""].copy()

    output_dir = Path(PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "audio_transcripts_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)

    print("\n--- Transcript Cleaning Report ---")
    print(f"Original rows: {original_count}")
    print(f"Empty transcripts: {empty_count}")
    print(f"Rows retained: {len(cleaned_df)}")
    print(f"Rows removed: {original_count - len(cleaned_df)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
