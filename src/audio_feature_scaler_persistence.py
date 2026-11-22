
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from config import PROCESSED_DIR

print("TRAINING-ONLY SCALER FIT AND PERSISTENCE")
print("=" * 55)

train_file = PROCESSED_DIR / "audio_features_train.csv"
validation_file = PROCESSED_DIR / "audio_features_validation.csv"
test_file = PROCESSED_DIR / "audio_features_test.csv"

train_df = pd.read_csv(train_file)
validation_df = pd.read_csv(validation_file)
test_df = pd.read_csv(test_file)

# Use numerical features only
train_numeric = train_df.select_dtypes(include="number")
validation_numeric = validation_df.select_dtypes(include="number")
test_numeric = test_df.select_dtypes(include="number")

# Fit scaler ONLY on training data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_numeric)

# Transform validation and test using the same fitted scaler
validation_scaled = scaler.transform(validation_numeric)
test_scaled = scaler.transform(test_numeric)

# Save scaled datasets
train_output = PROCESSED_DIR / "audio_features_train_standardized.csv"
validation_output = PROCESSED_DIR / "audio_features_validation_standardized.csv"
test_output = PROCESSED_DIR / "audio_features_test_standardized.csv"

pd.DataFrame(train_scaled, columns=train_numeric.columns).to_csv(
    train_output, index=False
)

pd.DataFrame(validation_scaled, columns=validation_numeric.columns).to_csv(
    validation_output, index=False
)

pd.DataFrame(test_scaled, columns=test_numeric.columns).to_csv(
    test_output, index=False
)

# Persist the fitted scaler for future reuse
scaler_file = PROCESSED_DIR / "audio_feature_standard_scaler.joblib"
joblib.dump(scaler, scaler_file)

print("\nSamples processed:")
print("Training:", len(train_scaled))
print("Validation:", len(validation_scaled))
print("Test:", len(test_scaled))

print("\nFiles saved:")
print(train_output)
print(validation_output)
print(test_output)
print(scaler_file)

print("\nSCALER FITTED ON TRAINING DATA ONLY")
print("SCALER PERSISTENCE COMPLETED")
