
from pathlib import Path
import soundfile as sf
import pandas as pd

from config import AUDIO_DIR, PROCESSED_DIR


def main():
    audio_dir = Path(AUDIO_DIR)

    if not audio_dir.exists():
        print(f"Audio directory not found: {audio_dir}")
        return

    audio_files = list(audio_dir.rglob("*.mp3"))
    audio_files += list(audio_dir.rglob("*.wav"))
    audio_files += list(audio_dir.rglob("*.flac"))

    print(f"Total audio files found: {len(audio_files)}")

    results = []

    for i, file_path in enumerate(audio_files, start=1):
        try:
            info = sf.info(str(file_path))

            results.append({
                "file_name": file_path.name,
                "file_path": str(file_path),
                "duration_seconds": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "format": info.format,
                "subtype": info.subtype,
                "file_size_kb": file_path.stat().st_size / 1024,
                "status": "OK"
            })

        except Exception as e:
            results.append({
                "file_name": file_path.name,
                "file_path": str(file_path),
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "format": None,
                "subtype": None,
                "file_size_kb": file_path.stat().st_size / 1024,
                "status": f"ERROR: {str(e)}"
            })

        if i % 100 == 0:
            print(f"Checked {i}/{len(audio_files)} files")

    df = pd.DataFrame(results)

    output_path = Path(PROCESSED_DIR) / "audio_dataset_integrity_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if not df.empty:
        valid_count = (df["status"] == "OK").sum()
        invalid_count = len(df) - valid_count

        print("\n--- Audio Integrity Report ---")
        print(f"Total files: {len(df)}")
        print(f"Valid files: {valid_count}")
        print(f"Problematic files: {invalid_count}")

        print("\nSample rate distribution:")
        print(df[df["status"] == "OK"]["sample_rate"].value_counts())

        print("\nChannel distribution:")
        print(df[df["status"] == "OK"]["channels"].value_counts())

        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
