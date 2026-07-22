import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Create folder
os.makedirs("data/processed", exist_ok=True)

# Load CKD dataset
df = pd.read_csv("data/ckd.csv")

# Split dataset
train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

# Save files
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

print("CKD files created successfully!")