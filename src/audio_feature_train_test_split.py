
import pandas as pd
from sklearn.model_selection import train_test_split
from config import PROCESSED_DIR

print("AUDIO FEATURE TRAIN / VALIDATION / TEST SPLIT")
print("=" * 55)

input_file = PROCESSED_DIR / "audio_features.csv"
df = pd.read_csv(input_file)

# First split: 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

# Second split: divide temporary data equally
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

# Save datasets
train_file = PROCESSED_DIR / "audio_features_train.csv"
validation_file = PROCESSED_DIR / "audio_features_validation.csv"
test_file = PROCESSED_DIR / "audio_features_test.csv"

train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)

print("\nDataset split results:")
print("Total samples:", len(df))
print("Training samples:", len(train_df))
print("Validation samples:", len(validation_df))
print("Test samples:", len(test_df))

print("\nSplit percentages:")
print(f"Training:   {len(train_df) / len(df) * 100:.2f}%")
print(f"Validation: {len(validation_df) / len(df) * 100:.2f}%")
print(f"Test:      {len(test_df) / len(df) * 100:.2f}%")

print("\nFiles saved:")
print(train_file)
print(validation_file)
print(test_file)

print("\nTRAIN / VALIDATION / TEST SPLIT COMPLETED")
