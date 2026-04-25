# predict_fertility.py
import joblib
import pandas as pd

# Load trained model
model = joblib.load("fertility_model.pkl")

# Example: User report input
# Replace these values with actual lab report parsing
user_report = {
    "Age": 21,
    "AMH": 5.5,
    "FSH": 2.0,
    "LH": 5.8,
    "Progesterone": 18,
    "Follicle_count": 19,
    "Ovarian_volume": 2,
    "Cycle_length": 28,
    "Cycle_regular": 0  # 1 = regular, 0 = irregular
}

# Convert to DataFrame
user_df = pd.DataFrame([user_report])

# Predict fertility
prediction = model.predict(user_df)[0]
print("Predicted Fertility Status:", prediction)

