import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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

# Replace missing zeros
missing_zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in missing_zero_columns:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Build MLP
model = Sequential([
    Dense(64, activation="relu", input_shape=(8,)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    verbose=1
)

# Plot Loss Curves
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/mlp_loss_curve.png")
plt.show()