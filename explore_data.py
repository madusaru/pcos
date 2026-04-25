import pandas as pd

# Load the ACTUAL PCOS dataset
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

# Remove junk unnamed columns (very common in medical datasets)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Shape of dataset:", df.shape)
print("\nColumn names:\n", df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values per column:")
print(df.isnull().sum())
