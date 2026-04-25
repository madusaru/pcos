import pytesseract
import cv2
import re
import pickle
import pandas as pd
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

model = None
if os.path.exists("male_infertility_model.pkl"):
    model = pickle.load(open("male_infertility_model.pkl", "rb"))
    print("✅ ML model loaded")
else:
    print("⚠ No ML model found → Using rules only")

# -------------------------------
# Load Image
# -------------------------------
image_path = r"C:\Users\HP\Desktop\pcos-project\semen.jpg"
image = cv2.imread(image_path)

if image is None:
    print("❌ Image not found. Check path.")
    exit()

# -------------------------------
# Preprocess Image
# -------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# -------------------------------
# OCR TEXT
# -------------------------------
text = pytesseract.image_to_string(thresh)

print("\n=== OCR TEXT ===\n")
print(text)

clean_text = text.lower().replace("\n", " ")

# -------------------------------
# Extract Values
# -------------------------------

# Volume
volume_match = re.search(r'volume\s*[:\-]?\s*(\d+\.?\d*)', clean_text)
volume = float(volume_match.group(1)) if volume_match else 0

# Sperm Count
count_match = re.search(r'sperm\s*count\s*[:\-]?\s*(\d+)', clean_text)
if count_match:
    sperm_count = float(count_match.group(1))
elif re.search(r'(nil|zero|absent)', clean_text):
    sperm_count = 0
else:
    sperm_count = 0

# Morphology
morph_match = re.search(r'(normal\s*forms|morphology)\s*[:\-]?\s*(\d+)', clean_text)
morphology = float(morph_match.group(2)) if morph_match else 0

# Motility
motility_match = re.search(r'motility\s*[:\-]?\s*(\d+)', clean_text)
motility = float(motility_match.group(1)) if motility_match else 0

# Age
age_match = re.search(r'age.*?(\d+)', clean_text)
age = int(age_match.group(1)) if age_match else 0

# Round values
volume = round(volume, 2)
sperm_count = round(sperm_count, 2)
motility = round(motility, 2)
morphology = round(morphology, 2)

# -------------------------------
# CONDITION DETECTION
# -------------------------------
if sperm_count == 0:
    detected_condition = "Azoospermia"

elif sperm_count < 15 and motility < 40 and morphology < 4:
    detected_condition = "Oligoasthenoteratozoospermia"

elif sperm_count < 15 and motility < 40:
    detected_condition = "Oligoasthenozoospermia"

elif sperm_count < 15 and morphology < 4:
    detected_condition = "Oligoteratozoospermia"

elif motility < 40 and morphology < 4:
    detected_condition = "Asthenoteratozoospermia"

elif sperm_count < 15:
    detected_condition = "Oligospermia"

elif motility < 40:
    detected_condition = "Asthenozoospermia"

elif morphology < 4:
    detected_condition = "Teratozoospermia"

else:
    detected_condition = "Normozoospermia"

# -------------------------------
# ML Prediction (optional)
# -------------------------------
if model is not None:
    try:
        input_data = pd.DataFrame([[age, volume, sperm_count, motility, morphology]],
                                 columns=['Age','Volume','SpermCount','Motility','Morphology'])
        ml_prediction = model.predict(input_data)[0]
    except:
        ml_prediction = "N/A"
else:
    ml_prediction = "N/A"

# -------------------------------
# RISK SCORE
# -------------------------------
risk_score = 0

if sperm_count < 15:
    risk_score += 40
if motility < 40:
    risk_score += 30
if morphology < 4:
    risk_score += 20
if volume < 1.5:
    risk_score += 5
if age > 40:
    risk_score += 5

if risk_score < 30:
    risk_level = "LOW"
elif risk_score < 60:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# -------------------------------
# FINAL REPORT OUTPUT
# -------------------------------
print("\n==============================")
print("🧾 SEMEN ANALYSIS REPORT")
print("==============================")

print(f"👤 Age: {age} years")
print(f"🧪 Semen Volume: {volume} ml (Normal: ≥ 1.5 ml)")
print(f"🔬 Sperm Count: {sperm_count} million/ml (Normal: ≥ 15)")
print(f"🏃 Motility: {motility}% (Normal: ≥ 40%)")
print(f"🧬 Morphology: {morphology}% (Normal: ≥ 4%)")

print("\n------------------------------")
print("🔍 PARAMETER ANALYSIS")
print("------------------------------")

if sperm_count == 0:
    print("❌ No sperm detected → Azoospermia")
elif sperm_count < 15:
    print("⚠ Low sperm count detected (Oligospermia risk)")
else:
    print("✅ Sperm count is normal")

if motility < 40:
    print("⚠ Poor sperm motility detected (Asthenozoospermia risk)")
else:
    print("✅ Motility is normal")

if morphology < 4:
    print("⚠ Abnormal sperm morphology detected (Teratozoospermia risk)")
else:
    print("✅ Morphology is normal")

if volume < 1.5:
    print("⚠ Low semen volume detected")

# ------------------------------
print("\n------------------------------")
print("🧬 FINAL DIAGNOSIS")
print("------------------------------")
print("Rule-based Diagnosis:", detected_condition)
print("ML Prediction:", ml_prediction)

# ------------------------------
print("\n------------------------------")
print("📊 RISK ASSESSMENT")
print("------------------------------")
print(f"Risk Score: {risk_score}%")
print(f"Risk Level: {risk_level}")

# ------------------------------
print("\n------------------------------")
print("💡 INTERPRETATION")
print("------------------------------")

if detected_condition == "Normozoospermia":
    print("✅ All semen parameters are within normal limits. Fertility is likely normal.")

elif "Azoospermia" in detected_condition:
    print("❗ No sperm detected. Requires immediate medical evaluation.")

elif "Oligo" in detected_condition:
    print("⚠ Low sperm count may reduce fertility chances.")

elif "Astheno" in detected_condition:
    print("⚠ Poor motility may affect sperm's ability to reach the egg.")

elif "Terato" in detected_condition:
    print("⚠ Abnormal morphology may affect fertilization capability.")

else:
    print("⚠ Mixed abnormalities detected affecting multiple parameters.")

# ------------------------------
print("\n==============================")
print("⚠ DISCLAIMER")
print("==============================")
print("This is an AI-generated report and not a medical diagnosis.")
print("Please consult a qualified doctor for professional advice.")

