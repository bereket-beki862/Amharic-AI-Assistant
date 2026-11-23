
from pathlib import Path
import pandas as pd
import soundfile as sf

from config import AUDIO_DIR, PROCESSED_DIR

MIN_DURATION = 1.0
MAX_DURATION = 30.0


def main():
    audio_dir = Path(AUDIO_DIR)
    extensions = {".wav", ".mp3", ".flac", ".ogg"}

    audio_files = sorted(
        file for file in audio_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in extensions
    )

    candidates = []
    excluded = []

    for file_path in audio_files:
        try:
            info = sf.info(str(file_path))
            duration = info.duration

            record = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "duration_seconds": round(duration, 3),
                "sample_rate": info.samplerate,
                "channels": info.channels
            }

            if MIN_DURATION <= duration <= MAX_DURATION:
                candidates.append(record)
            else:
                excluded.append(record)

        except Exception as error:
            print(f"Could not read {file_path.name}: {error}")

    candidates_df = pd.DataFrame(candidates)
    excluded_df = pd.DataFrame(excluded)

    output_dir = Path(PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / "whisper_audio_candidates.csv"
    excluded_path = output_dir / "whisper_audio_excluded.csv"

    candidates_df.to_csv(candidates_path, index=False)
    excluded_df.to_csv(excluded_path, index=False)

    print("\n--- Whisper Audio Candidate Selection ---")
    print(f"Total audio files: {len(audio_files)}")
    print(f"Selected (1–30 sec): {len(candidates)}")
    print(f"Excluded: {len(excluded)}")
    print(f"\nCandidates saved: {candidates_path}")
    print(f"Excluded files saved: {excluded_path}")


if __name__ == "__main__":
    main()
