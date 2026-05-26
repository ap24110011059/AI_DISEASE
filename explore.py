import pandas as pd

# Load CSV file
df = pd.read_csv("data.csv")

# Print shape
print("Shape of dataset:")
print(df.shape)

# Print data types
print("\nData Types:")
print(df.dtypes)

# Print statistical summary
print("\nSummary Statistics:")
print(df.describe())