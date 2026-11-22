
import pandas as pd
from config import PROCESSED_DIR

print("TRAIN / VALIDATION / TEST DATA LEAKAGE CHECK")
print("=" * 60)

train_file = PROCESSED_DIR / "audio_features_train.csv"
validation_file = PROCESSED_DIR / "audio_features_validation.csv"
test_file = PROCESSED_DIR / "audio_features_test.csv"

train_df = pd.read_csv(train_file)
validation_df = pd.read_csv(validation_file)
test_df = pd.read_csv(test_file)

# Convert rows into comparable string representations
train_rows = set(map(tuple, train_df.astype(str).values))
validation_rows = set(map(tuple, validation_df.astype(str).values))
test_rows = set(map(tuple, test_df.astype(str).values))

train_validation = train_rows & validation_rows
train_test = train_rows & test_rows
validation_test = validation_rows & test_rows

print("\nDataset sizes:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Test:", len(test_df))

print("\nExact duplicate rows across splits:")
print("Train vs Validation:", len(train_validation))
print("Train vs Test:", len(train_test))
print("Validation vs Test:", len(validation_test))

leakage_found = (
    len(train_validation) > 0
    or len(train_test) > 0
    or len(validation_test) > 0
)

print("\nLeakage status:")
if leakage_found:
    print("WARNING: Identical rows detected across dataset splits.")
else:
    print("PASS: No identical rows detected across dataset splits.")

output_file = PROCESSED_DIR / "audio_feature_split_leakage_report.csv"

report = pd.DataFrame({
    "comparison": [
        "Train vs Validation",
        "Train vs Test",
        "Validation vs Test"
    ],
    "identical_rows": [
        len(train_validation),
        len(train_test),
        len(validation_test)
    ]
})

report.to_csv(output_file, index=False)

print("\nReport saved:", output_file)
print("LEAKAGE CHECK COMPLETED")
