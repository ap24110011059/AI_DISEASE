import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve

# Load adaptive model results
df = pd.read_csv("results/adaptive_results.csv")

# -----------------------------------
# Rolling Window Features (Last 3 Batches)
# -----------------------------------

df["rolling_mean"] = df["Accuracy"].rolling(window=3).mean()
df["rolling_std"] = df["Accuracy"].rolling(window=3).std()

# Trend = current mean - previous mean
df["trend"] = df["rolling_mean"].diff()

# -----------------------------------
# Predict Risk 2 Steps Ahead
# -----------------------------------

df["future_accuracy"] = df["Accuracy"].shift(-2)

# Risk if future accuracy drops below 0.75
df["future_risk"] = (df["future_accuracy"] < 0.75).astype(int)

# -----------------------------------
# Alert Threshold Tuning
# -----------------------------------

scores = 1 - df["rolling_mean"].fillna(1)

precision, recall, thresholds = precision_recall_curve(
    df["future_risk"].fillna(0),
    scores
)

best_threshold = 0.50

if len(thresholds) > 0:
    f1 = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-8
    )
    best_threshold = thresholds[np.argmax(f1)]

# -----------------------------------
# Generate Alerts
# -----------------------------------

df["Alert"] = (scores >= best_threshold).astype(int)

# -----------------------------------
# Save Results
# -----------------------------------

df.to_csv(
    "results/proactive_results.csv",
    index=False
)

print(df)

print("\nBest Alert Threshold:", round(best_threshold,3))
print("Saved to results/proactive_results.csv")