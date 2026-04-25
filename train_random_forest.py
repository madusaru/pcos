import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

# -----------------------------
# Drop ID columns
# -----------------------------
df.drop(columns=["Sl. No", "Patient File No."], inplace=True)

# -----------------------------
# Handle missing values
# -----------------------------
df["Marraige Status (Yrs)"] = df["Marraige Status (Yrs)"].fillna(
    df["Marraige Status (Yrs)"].median()
)

df["Fast food (Y/N)"] = df["Fast food (Y/N)"].fillna(
    df["Fast food (Y/N)"].mode()[0]
)

# -----------------------------
# Split features & target
# -----------------------------
y = df["PCOS (Y/N)"]
X = df.drop("PCOS (Y/N)", axis=1)

# ⭐ PRINT FEATURES (VERY IMPORTANT)
print("\n================ MODEL FEATURES ================\n")
print(X.columns.tolist())
print("\n================================================\n")

# -----------------------------
# Ensure all numeric
# -----------------------------
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
joblib.dump(rf, "model/pcos_random_forest.pkl")
print("\n✅ Random Forest model saved successfully")


feature_info = {
    "features": X.columns.tolist(),
    "medians": X.median().to_dict()
}

with open("model/feature_info.json", "w") as f:
    json.dump(feature_info, f)

print("✅ Feature info saved")