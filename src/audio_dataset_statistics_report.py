
from pathlib import Path
import soundfile as sf
import pandas as pd

from config import AUDIO_DIR, PROCESSED_DIR


def main():
    audio_dir = Path(AUDIO_DIR)

    audio_files = list(audio_dir.rglob("*.mp3"))
    audio_files += list(audio_dir.rglob("*.wav"))
    audio_files += list(audio_dir.rglob("*.flac"))

    results = []

    for file_path in audio_files:
        try:
            info = sf.info(str(file_path))

            results.append({
                "file_name": file_path.name,
                "duration_seconds": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "file_size_mb": file_path.stat().st_size / (1024 * 1024)
            })

        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    df = pd.DataFrame(results)

    print("\n========== AUDIO DATASET STATISTICS ==========")
    print(f"Total readable audio files: {len(df)}")

    if not df.empty:
        print(f"Total duration (hours): {df['duration_seconds'].sum() / 3600:.2f}")
        print(f"Average duration (seconds): {df['duration_seconds'].mean():.2f}")
        print(f"Median duration (seconds): {df['duration_seconds'].median():.2f}")
        print(f"Total storage (MB): {df['file_size_mb'].sum():.2f}")
        print(f"Average file size (MB): {df['file_size_mb'].mean():.2f}")

        print("\nSample rate distribution:")
        print(df["sample_rate"].value_counts())

        print("\nChannel distribution:")
        print(df["channels"].value_counts())

        print("\nDuration statistics:")
        print(df["duration_seconds"].describe())

    output_path = Path(PROCESSED_DIR) / "audio_dataset_statistics_report.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
