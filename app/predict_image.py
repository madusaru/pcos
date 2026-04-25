import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ================= CONFIG =================
MODEL_PATH = "pcos_mobilenet_model.h5"
IMAGE_PATH = "st1.jpg"

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# ================= LOAD IMAGE =================
img = image.load_img(IMAGE_PATH, target_size=(224, 224))

# Convert to RGB if needed
if img.mode != "RGB":
    img = img.convert("RGB")

# Convert image to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Expand dimensions for model
img_array = np.expand_dims(img_array, axis=0)

# ================= SHOW IMAGE =================
plt.imshow(img)
plt.title("Input Image")
plt.axis("off")
plt.show()

# ================= PREDICTION =================
prediction = model.predict(img_array)[0][0]

print("\nRaw model output:", prediction)
print(f"Model confidence score: {prediction*100:.2f}%")

# ================= RESULT =================
# (class order might be reversed depending on dataset)

if prediction > 0.5:
    print("RESULT: NO PCOS DETECTED")
else:
    print("RESULT: PCOS DETECTED")

