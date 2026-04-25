import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

# Drop ID columns
df.drop(columns=["Sl. No", "Patient File No."], inplace=True)

# Fix missing values (safe way)
df["Marraige Status (Yrs)"] = df["Marraige Status (Yrs)"].fillna(
    df["Marraige Status (Yrs)"].median()
)
df["Fast food (Y/N)"] = df["Fast food (Y/N)"].fillna(
    df["Fast food (Y/N)"].mode()[0]
)

# Separate target
y = df["PCOS (Y/N)"]
X = df.drop("PCOS (Y/N)", axis=1)

# 🔥 KEY FIX: convert everything to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Handle any new NaNs created by conversion
X = X.fillna(X.median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)
print("\nTrain target distribution:")
print(y_train.value_counts())
