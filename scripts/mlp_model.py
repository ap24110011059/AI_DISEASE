import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

# Replace missing zeros with median
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

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Reduce Learning Rate Callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

batch_sizes = [32, 64, 128]

best_accuracy = 0
best_batch = None
best_history = None
best_metrics = None

for batch_size in batch_sizes:

    print(f"\nTraining with Batch Size = {batch_size}")

    # Build Model
    model = Sequential([
        Dense(64, activation="relu", input_shape=(8,)),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=batch_size,
        callbacks=[reduce_lr],
        verbose=1
    )

    # Predictions
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_batch = batch_size
        best_history = history

        best_metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc
        }

# Save Loss Curve of Best Model
plt.figure(figsize=(8,5))

plt.plot(best_history.history["loss"], label="Training Loss")
plt.plot(best_history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/mlp_loss_curve.png")
plt.show()

# Save Metrics
results = pd.DataFrame({
    "model": ["MLP"],
    "accuracy": [best_metrics["accuracy"]],
    "precision": [best_metrics["precision"]],
    "recall": [best_metrics["recall"]],
    "f1": [best_metrics["f1"]],
    "auc_roc": [best_metrics["auc_roc"]]
})

results.to_csv(
    "results/results_log.csv",
    mode="a",
    header=False,
    index=False
)

# Save Best Config
best_config = {
    "dropout": 0.3,
    "optimizer": "Adam",
    "epochs": 50,
    "best_batch_size": best_batch,
    "best_accuracy": float(best_accuracy)
}

with open("best_config.json", "w") as f:
    json.dump(best_config, f, indent=4)

print("\nBest Configuration:")
print(best_config)

print("\nMLP Metrics:")
print(results)