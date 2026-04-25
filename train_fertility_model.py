# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # to save/load model

# Load dataset
df = pd.read_csv("female_fertility_dataset_realistic.csv")

# Features and labels
X = df.drop("Fertility_label", axis=1)
y = df["Fertility_label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "fertility_model.pkl")
print("Model saved as fertility_model.pkl")

