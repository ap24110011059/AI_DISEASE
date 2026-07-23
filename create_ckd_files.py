import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/ckd.csv")

train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

df.to_csv(
    "data/processed/chronic_kidney_disease_processed.csv",
    index=False
)

train.to_csv(
    "data/processed/chronic_kidney_disease_train.csv",
    index=False
)

test.to_csv(
    "data/processed/chronic_kidney_disease_test.csv",
    index=False
)

print("All CKD files created successfully!")