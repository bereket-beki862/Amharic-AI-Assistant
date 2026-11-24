
from pathlib import Path
import pandas as pd

from config import AUDIO_DIR, PROCESSED_DIR


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    tsv_files = list(raw_dir.rglob("train.tsv"))

    if not tsv_files:
        print("train.tsv not found in data/raw")
        return

    transcript_file = tsv_files[0]
    print(f"Using transcript file: {transcript_file}")

    transcripts = pd.read_csv(transcript_file, sep="\t")

    audio_dir = Path(AUDIO_DIR)
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg"}

    audio_index = {}

    for file_path in audio_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_index.setdefault(file_path.name, []).append(str(file_path))

    matched = []
    missing = []
    duplicates = []

    for _, row in transcripts.iterrows():
        filename = str(row.get("path", ""))
        sentence = row.get("sentence", "")

        matches = audio_index.get(filename, [])

        if len(matches) == 1:
            matched.append({
                "file_name": filename,
                "audio_path": matches[0],
                "sentence": sentence
            })
        elif len(matches) == 0:
            missing.append({
                "file_name": filename,
                "sentence": sentence
            })
        else:
            duplicates.append({
                "file_name": filename,
                "sentence": sentence,
                "matching_paths": " | ".join(matches)
            })

    output_dir = Path(PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    matched_df = pd.DataFrame(matched)
    missing_df = pd.DataFrame(missing)
    duplicates_df = pd.DataFrame(duplicates)

    matched_df.to_csv(output_dir / "audio_transcript_matched.csv", index=False)
    missing_df.to_csv(output_dir / "audio_transcript_missing.csv", index=False)
    duplicates_df.to_csv(output_dir / "audio_transcript_duplicates.csv", index=False)

    print("\n--- Transcript Matching Report ---")
    print(f"Transcript rows: {len(transcripts)}")
    print(f"Matched: {len(matched)}")
    print(f"Missing audio: {len(missing)}")
    print(f"Multiple matches: {len(duplicates)}")
    print(f"Reports saved in: {output_dir}")


if __name__ == "__main__":
    main()
