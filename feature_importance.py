import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_excel(
    "data/PCOS_data_without_infertility.xlsx",
    sheet_name="Full_new"
)

df.drop(columns=["Sl. No", "Patient File No."], inplace=True)

# Handle missing values
df["Marraige Status (Yrs)"] = df["Marraige Status (Yrs)"].fillna(
    df["Marraige Status (Yrs)"].median()
)
df["Fast food (Y/N)"] = df["Fast food (Y/N)"].fillna(
    df["Fast food (Y/N)"].mode()[0]
)

y = df["PCOS (Y/N)"]
X = df.drop("PCOS (Y/N)", axis=1)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X, y)

# Feature importance
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Plot top 15
importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 Important Features for PCOS Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
