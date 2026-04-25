import pandas as pd

# Load data
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

# Drop ID-like columns (not useful for ML)
df.drop(columns=["Sl. No", "Patient File No."], inplace=True)

# Handle missing values
df["Marraige Status (Yrs)"] = df["Marraige Status (Yrs)"].fillna(
    df["Marraige Status (Yrs)"].median()
)

df["Fast food (Y/N)"] = df["Fast food (Y/N)"].fillna(
    df["Fast food (Y/N)"].mode()[0]
)


# Separate features & target
X = df.drop("PCOS (Y/N)", axis=1)
y = df["PCOS (Y/N)"]

print("After cleaning:")
print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts())
