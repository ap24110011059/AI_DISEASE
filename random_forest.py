import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

RANDOM_SEED = 42

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

df = pd.read_csv("pima.csv", names=columns)

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

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

feature_names = X.columns

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_SEED
)

# Grid Search
param_grid = {
    "n_estimators": [100, 200, 500]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_SEED),
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# Predictions
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Append to results file
results = pd.DataFrame({
    "model": ["RF"],
    "accuracy": [accuracy],
    "precision": [precision],
    "recall": [recall],
    "f1": [f1],
    "auc_roc": [auc]
})

results.to_csv(
    "results/results_log.csv",
    mode="a",
    header=False,
    index=False
)

print(results)

# Feature Importance
importance = best_model.feature_importances_

plt.figure(figsize=(8,5))
plt.bar(feature_names, importance)
plt.xticks(rotation=45)
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig("results/rf_feature_importance.png")
plt.show()
