import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

# -----------------------------
# Select ONLY simple features
# -----------------------------
FEATURES = [
    " Age (yrs)",
    "Weight (Kg)",
    "Height(Cm) ",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Hair loss(Y/N)",
    "Pimples(Y/N)",
    "Skin darkening (Y/N)",
    "Reg.Exercise(Y/N)",
    "Fast food (Y/N)"
]

TARGET = "PCOS (Y/N)"

df = df[FEATURES + [TARGET]]

# -----------------------------
# Convert to numeric
# -----------------------------
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median())

X = df[FEATURES]
y = df[TARGET]

print("\nMODEL FEATURES USED:\n", X.columns.tolist())

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "model/pcos_random_forest.pkl")

feature_info = {
    "features": FEATURES,
    "medians": X.median().to_dict()
}

with open("model/feature_info.json", "w") as f:
    json.dump(feature_info, f)

print("\n✅ Simple PCOS model trained & saved successfully")
