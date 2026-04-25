import pickle
import numpy as np

# Load trained model
model = pickle.load(open("male_infertility_model.pkl", "rb"))

# Example patient values
# Age, Volume, Abstinence, Count, Active Motile, Weakly Motile,
# Non Motile, Morphology Normal, Morphology Abnormal
input_data = np.array([[30, 2.5, 3, 60, 40, 30, 30, 60, 40]])

# Predict
prediction = model.predict(input_data)

print("Predicted Condition:", prediction[0])

