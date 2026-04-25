# ---------------------------------------
# PCOS COMPLETE SYSTEM
# 1. Symptom-based risk check
# 2. Image-based PCOS classification
# ---------------------------------------

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# -----------------------------
# PATHS & SETTINGS
# -----------------------------
MODEL_PATH = "best_model.h5"   # MAKE SURE THIS FILE EXISTS
IMG_SIZE = 224
THRESHOLD = 0.3   # LOWERED for medical sensitivity

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading PCOS image model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully ✅")

# -----------------------------
# STEP 1: SYMPTOM CHECK
# -----------------------------
def symptom_check():
    print("\n--- PCOS SYMPTOM CHECK ---")

    irregular_periods = int(input("Irregular periods? (1 = Yes, 0 = No): "))
    weight_gain = int(input("Weight gain? (1 = Yes, 0 = No): "))
    acne = int(input("Acne? (1 = Yes, 0 = No): "))
    hair_growth = int(input("Excess hair growth? (1 = Yes, 0 = No): "))

    score = irregular_periods + weight_gain + acne + hair_growth

    if score >= 2:
        print("\n⚠️ PCOS RISK DETECTED (Proceeding to image analysis)")
        return True
    else:
        print("\n✅ LOW PCOS RISK")
        return False

# -----------------------------
# STEP 2: IMAGE PREDICTION
# -----------------------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print("❌ Image file not found!")
        return

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    print("\n--- IMAGE RESULT ---")
    print(f"PCOS Probability: {prediction:.2f}")

    if prediction >= THRESHOLD:
        print("🩺 FINAL RESULT: PCOS DETECTED")
    else:
        print("✅ FINAL RESULT: NON-PCOS")

# -----------------------------
# MAIN FLOW
# -----------------------------
if __name__ == "__main__":
    risk = symptom_check()

    if risk:
        img_path = input("\nEnter ultrasound image path: ")
        predict_image(img_path)
    else:
        print("\nImage test not required.")
