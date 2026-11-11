
import pandas as pd
import whisper

from pathlib import Path
from config import PROJECT_ROOT

# Load test dataset
test_file = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"
df = pd.read_csv(test_file)

# Load Whisper Small
print("Loading Whisper Small...")
model = whisper.load_model("small", device="cpu")

results = []

# Evaluate 5 samples
for i in range(min(5, len(df))):

    audio_path = PROJECT_ROOT / df.loc[i, "audio_path"]
    reference = df.loc[i, "sentence"]

    print(f"\nProcessing sample {i + 1}/5...")

    prediction = model.transcribe(
        str(audio_path),
        language="am",
        fp16=False
    )["text"].strip()

    print("REFERENCE:", reference)
    print("PREDICTION:", prediction)

    results.append({
        "audio_path": df.loc[i, "audio_path"],
        "reference": reference,
        "prediction": prediction
    })

# Save evaluation results
output_file = PROJECT_ROOT / "data" / "processed" / "whisper_evaluation.csv"
pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")

print("\nEVALUATION COMPLETED")
print("Saved to:", output_file)
