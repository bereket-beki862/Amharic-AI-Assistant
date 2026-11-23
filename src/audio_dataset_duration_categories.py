
from pathlib import Path
import soundfile as sf
import pandas as pd

from config import AUDIO_DIR, PROCESSED_DIR


def categorize_duration(duration):
    if duration < 3:
        return "Short"
    elif duration <= 10:
        return "Medium"
    else:
        return "Long"


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
                "duration_category": categorize_duration(info.duration)
            })

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    df = pd.DataFrame(results)

    print("\n--- Audio Duration Categories ---")
    print(f"Total readable files: {len(df)}")

    if not df.empty:
        print("\nCategory counts:")
        print(df["duration_category"].value_counts())

        print("\nAverage duration:")
        print(f"{df['duration_seconds'].mean():.2f} seconds")

        print("\nDuration statistics:")
        print(df["duration_seconds"].describe())

    output_path = Path(PROCESSED_DIR) / "audio_duration_categories.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
