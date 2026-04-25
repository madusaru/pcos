import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

# Drop identifiers with no predictive value
df.drop(columns=["Sl. No", "Patient File No."], inplace=True)

# Fix missing values
df["Marraige Status (Yrs)"] = df["Marraige Status (Yrs)"].fillna(df["Marraige Status (Yrs)"].median())
df["Fast food (Y/N)"] = df["Fast food (Y/N)"].fillna(df["Fast food (Y/N)"].mode()[0])

# Separate target
y = df["PCOS (Y/N)"]
X = df.drop("PCOS (Y/N)", axis=1)

# Properly encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Convert numeric safely and fill remaining NaNs
X = X.apply(pd.to_numeric, errors="coerce").fillna(X.median())

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# === Random Forest Model ===
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test_scaled)

print("=== Random Forest Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC‑AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:,1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# === XGBoost Model ===
xgb = XGBClassifier(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=5,
    subsample=0.8,
    scale_pos_weight=1,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)
xgb.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb.predict(X_test_scaled)

print("=== XGBoost Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC‑AUC:", roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:,1]))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))
