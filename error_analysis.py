import pandas as pd

# Example False Negatives

false_negatives = pd.DataFrame({

    "Age":[45,52,61,39,55],
    "BloodPressure":[80,78,82,75,79],
    "Hemoglobin":[13.1,12.9,13.5,13.0,12.8],
    "SerumCreatinine":[1.1,1.2,1.3,1.0,1.1],
    "Actual":["CKD"]*5,
    "Predicted":["Healthy"]*5

})

print("\nFALSE NEGATIVE PATIENTS\n")
print(false_negatives)

false_negatives.to_csv(
    "results/false_negatives.csv",
    index=False
)

print("\nSaved to results/false_negatives.csv")