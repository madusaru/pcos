import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ================= CONFIG =================
MODEL_PATH = "best_model.h5"
IMAGE_PATH = "st1.jpg"

HIGH_CONF_THRESHOLD = 0.85
LOW_CONF_THRESHOLD  = 0.50

# ================= SAFETY CHECKS =================
if not os.path.isfile(MODEL_PATH):
    print(f"❌ Model not found at: {MODEL_PATH}")
    exit()

if not os.path.isfile(IMAGE_PATH):
    print(f"❌ Image not found at: {IMAGE_PATH}")
    exit()

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# ================= LOAD & PREPROCESS IMAGE =================
img = image.load_img(IMAGE_PATH, target_size=(224, 224))

# Convert grayscale → RGB if needed
if img.mode != "RGB":
    img = img.convert("RGB")

img_array = image.img_to_array(img).astype("float32")

# 🔥 Ultrasound-safe per-image standardization
mean = np.mean(img_array)
std = np.std(img_array) + 1e-7
img_array = (img_array - mean) / std

img_array = np.expand_dims(img_array, axis=0)

# ================= DISPLAY PREPROCESSED IMAGE =================
plt.imshow(
    (img_array[0] - img_array[0].min()) /
    (img_array[0].max() - img_array[0].min())
)
plt.title("Preprocessed Ultrasound Image")
plt.axis("off")
plt.show()

# ================= RUN PREDICTION =================
prediction = float(model.predict(img_array)[0][0])
confidence = prediction * 100

print("\n================ AI ANALYSIS OUTPUT ================")
print(f"Model confidence score : {confidence:.2f}%")

# ================= INTERPRETATION =================
if prediction >= HIGH_CONF_THRESHOLD:
    print("\n🔴 Interpretation:")
    print("High presence of visual patterns commonly associated with PCOS.")
    print("Result classification : PCOS-related features detected")

elif prediction >= LOW_CONF_THRESHOLD:
    print("\n🟠 Interpretation:")
    print("Moderate presence of overlapping ovarian patterns.")
    print("Result classification : Inconclusive / Mixed visual features")

else:
    print("\n🟢 Interpretation:")
    print("Low presence of PCOS-associated visual patterns.")
    print("Result classification : No significant PCOS-related features detected")

# ================= DISCLAIMER =================
print("\n⚠ DISCLAIMER")
print("This system is an AI-assisted research tool.")
print("It does NOT provide a medical diagnosis.")
print("Final diagnosis must be made by a qualified healthcare professional.")
print("====================================================")
