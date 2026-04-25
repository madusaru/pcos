# predict_from_image.py
import pytesseract
from PIL import Image
import re
import pandas as pd
import joblib

# Configure Tesseract path (Windows example)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load trained model
model = joblib.load("fertility_model.pkl")

# Load lab report image
image_path = "Infertility_102.png"  # replace with actual image path
img = Image.open(image_path)

# Extract text using OCR
text = pytesseract.image_to_string(img)
print("Extracted Text from Report:\n", text)

# ---------------------------
# Step 2: Parse hormone values
# ---------------------------
def extract_value(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0  # default 0.0 if not found

# Example regex patterns (adjust depending on report format)
amh = extract_value(r"AMH\s*[:=]?\s*([0-9.]+)", text)
fsh = extract_value(r"FSH\s*[:=]?\s*([0-9.]+)", text)
lh = extract_value(r"LH\s*[:=]?\s*([0-9.]+)", text)
prog = extract_value(r"Progesterone\s*[:=]?\s*([0-9.]+)", text)
follicle_count = extract_value(r"Follicle Count\s*[:=]?\s*([0-9]+)", text)
ovarian_volume = extract_value(r"Ovarian Volume\s*[:=]?\s*([0-9.]+)", text)
cycle_length = extract_value(r"Cycle Length\s*[:=]?\s*([0-9]+)", text)
cycle_regular = extract_value(r"Cycle Regular\s*[:=]?\s*([01])", text)
age = extract_value(r"Age\s*[:=]?\s*([0-9]+)", text)

# ---------------------------
# Step 3: Create DataFrame
# ---------------------------
user_df = pd.DataFrame([{
    "Age": age,
    "AMH": amh,
    "FSH": fsh,
    "LH": lh,
    "Progesterone": prog,
    "Follicle_count": follicle_count,
    "Ovarian_volume": ovarian_volume,
    "Cycle_length": cycle_length,
    "Cycle_regular": cycle_regular
}])

# ---------------------------
# Step 4: Predict fertility
# ---------------------------
prediction = model.predict(user_df)[0]
print("\nPredicted Fertility Status:", prediction)

