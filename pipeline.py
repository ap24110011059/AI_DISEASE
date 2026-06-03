import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Fixed random seed
RANDOM_SEED = 42

# Column names
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# Load dataset
df = pd.read_csv("pima.csv", names=columns)

# Columns where 0 means missing value
missing_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

# Replace zeros with median
for col in missing_zero_columns:
    median_value = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_value)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Min-Max Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)

# Output
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nClass Distribution:")
print(y_train.value_counts())
print(y_test.value_counts())