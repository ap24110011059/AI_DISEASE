import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------
# Load Dataset
# -----------------------------
columns = [
    "Pregnancies","Glucose","BloodPressure",
    "SkinThickness","Insulin","BMI",
    "DiabetesPedigreeFunction","Age","Outcome"
]

df = pd.read_csv("pima.csv", names=columns)

# Replace missing zeros
missing_cols = [
    "Glucose","BloodPressure",
    "SkinThickness","Insulin","BMI"
]

for col in missing_cols:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)

# -----------------------------
# Normalize
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Create 10 Sequential Batches
# -----------------------------
batch_size = len(X) // 10

results = []

for i in range(10):

    start = i * batch_size

    if i == 9:
        end = len(X)
    else:
        end = (i + 1) * batch_size

    X_batch = X[start:end]
    y_batch = y.iloc[start:end]

    if len(X_batch) < 20:
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X_batch,
        y_batch,
        test_size=0.20,
        random_state=42
    )

    # -----------------------------
    # Adaptive MLP
    # -----------------------------
    model = Sequential([
        Dense(64, activation="relu", input_shape=(8,)),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=10,
        verbose=0
    )

    # Prediction
    probs = model.predict(X_test, verbose=0)

    threshold = 0.50

    preds = (probs > threshold).astype(int)

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Batch": i + 1,
        "Accuracy": round(accuracy,3),
        "F1": round(f1,3),
        "Threshold": threshold
    })

# -----------------------------
# Save Results
# -----------------------------
results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/adaptive_results.csv",
    index=False
)

print(results_df)

print("\nSaved to results/adaptive_results.csv")