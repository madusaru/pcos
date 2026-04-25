from flask import Flask, render_template, request, session
import joblib
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import pytesseract
import re

app = Flask(__name__)
app.secret_key = "secret123"

# -------------------------
# TESSERACT PATH
# -------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# -------------------------
# LOAD MODELS
# -------------------------
rf_model = joblib.load(os.path.join(MODEL_DIR, "pcos_random_forest.pkl"))
cnn_model = load_model(os.path.join(MODEL_DIR, "pcos_mobilenet_model.h5"))

# ❗ SAFE LOAD (NO CRASH)
fertility_path = os.path.join(MODEL_DIR, "fertility_model.pkl")
if os.path.exists(fertility_path):
    fertility_model = joblib.load(fertility_path)
    print("✅ Fertility model loaded")
else:
    fertility_model = None
    print("⚠️ Fertility model NOT found (disabled)")

# -------------------------
# FEATURE MAPPING
# -------------------------
FEATURE_ORDER = [
    " Age (yrs)", "Weight (Kg)", "Height(Cm) ",
    "Cycle(R/I)", "Cycle length(days)",
    "Weight gain(Y/N)", "hair growth(Y/N)",
    "Hair loss(Y/N)", "Pimples(Y/N)",
    "Skin darkening (Y/N)", "Reg.Exercise(Y/N)",
    "Fast food (Y/N)"
]

FORM_TO_FEATURE = {
    "age": " Age (yrs)",
    "weight": "Weight (Kg)",
    "height": "Height(Cm) ",
    "cycle_regular": "Cycle(R/I)",
    "cycle_length": "Cycle length(days)",
    "weight_gain": "Weight gain(Y/N)",
    "hair_growth": "hair growth(Y/N)",
    "hair_loss": "Hair loss(Y/N)",
    "pimples": "Pimples(Y/N)",
    "skin_darkening": "Skin darkening (Y/N)",
    "exercise": "Reg.Exercise(Y/N)",
    "fast_food": "Fast food (Y/N)"
}

def yes_no_to_int(value):
    return 1 if str(value).lower() == "yes" else 0

# -------------------------
# HOME
# -------------------------
@app.route("/")
def home():
    return render_template("home.html")

# -------------------------
# FEMALE FORM
# -------------------------
@app.route("/female")
def female():
    return render_template("female_form.html")

# -------------------------
# STEP 1: PCOS
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    form_data = {}

    for key, feature in FORM_TO_FEATURE.items():
        val = request.form.get(key, "0")

        try:
            form_data[feature] = yes_no_to_int(val) if val.lower() in ["yes","no"] else float(val)
        except:
            form_data[feature] = 0

    # STORE DATA
    session["pcos_data"] = form_data

    input_df = pd.DataFrame([form_data])[FEATURE_ORDER]

    prediction = rf_model.predict(input_df)[0]
    probability = round(rf_model.predict_proba(input_df)[0][1]*100,2)

    if prediction == 1:
        return render_template("upload.html", probability=probability)
    else:
        return render_template("result.html",
                               result="PCOS NOT LIKELY",
                               probability=probability)

# -------------------------
# STEP 2: ULTRASOUND
# -------------------------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files.get("image")

    if not file or file.filename == "":
        return "No image uploaded"

    path = os.path.join(BASE_DIR, "temp.jpg")
    file.save(path)

    img = image.load_img(path, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = cnn_model.predict(img_array)[0][0]
    os.remove(path)

    if pred < 0.5:
        result = "PCOS DETECTED (Ultrasound)"
        confidence = round((1-pred)*100,2)
    else:
        result = "NO PCOS DETECTED (Ultrasound)"
        confidence = round(pred*100,2)

    return render_template("after_ultrasound.html",
                           result=result,
                           probability=confidence)

# -------------------------
# STEP 3: FERTILITY PAGE
# -------------------------
@app.route("/fertility")
def fertility():
    return render_template("fertility_upload.html")

# -------------------------
# STEP 4: FERTILITY
# -------------------------
@app.route("/predict_fertility", methods=["POST"])
def predict_fertility():

    # ❗ MODEL NOT READY
    if fertility_model is None:
        return "⚠️ Fertility model not trained yet"

    file = request.files.get("image")

    if not file or file.filename == "":
        return "No file uploaded"

    path = os.path.join(BASE_DIR, "fertility.jpg")
    file.save(path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray).lower()

    def extract(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0

    # Extract values
    lh = extract(r'lh[:\-]?\s*(\d+\.?\d*)')
    fsh = extract(r'fsh[:\-]?\s*(\d+\.?\d*)')
    amh = extract(r'amh[:\-]?\s*(\d+\.?\d*)')
    prog = extract(r'progesterone[:\-]?\s*(\d+\.?\d*)')
    follicle = extract(r'follicle[:\-]?\s*(\d+)')
    ovary = extract(r'ovarian volume[:\-]?\s*(\d+\.?\d*)')

    pcos_data = session.get("pcos_data", {})

    features = [
        pcos_data.get(" Age (yrs)", 0),
        pcos_data.get("Weight (Kg)", 0),
        pcos_data.get("Cycle length(days)", 0),
        pcos_data.get("Cycle(R/I)", 0),
        pcos_data.get("Weight gain(Y/N)", 0),
        pcos_data.get("hair growth(Y/N)", 0),
        pcos_data.get("Hair loss(Y/N)", 0),
        pcos_data.get("Pimples(Y/N)", 0),
        pcos_data.get("Skin darkening (Y/N)", 0),
        pcos_data.get("Reg.Exercise(Y/N)", 0),
        pcos_data.get("Fast food (Y/N)", 0),

        lh, fsh, amh, prog, follicle, ovary
    ]

    pred = fertility_model.predict([features])[0]

    result = "FERTILE" if pred == 1 else "INFERTILE"

    os.remove(path)

    return render_template("female_result.html", result=result)

# -------------------------
# MALE
# -------------------------
@app.route("/male")
def male():
    return render_template("male_upload.html")

@app.route("/predict_male", methods=["POST"])
def predict_male():
    file = request.files.get("image")

    if not file:
        return "No file uploaded"

    path = os.path.join(BASE_DIR, "male_temp.jpg")
    file.save(path)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(gray).lower()

    sperm = 0
    motility = 0
    morphology = 0

    sc = re.search(r'sperm\s*count[:\-]?\s*(\d+)', text)
    mot = re.search(r'motility[:\-]?\s*(\d+)', text)
    mor = re.search(r'morphology[:\-]?\s*(\d+)', text)

    if sc: sperm = float(sc.group(1))
    if mot: motility = float(mot.group(1))
    if mor: morphology = float(mor.group(1))

    if sperm == 0:
        condition = "Azoospermia"
    elif sperm < 15:
        condition = "Oligospermia"
    elif motility < 40:
        condition = "Asthenozoospermia"
    elif morphology < 4:
        condition = "Teratozoospermia"
    else:
        condition = "Normozoospermia"

    os.remove(path)

    return render_template("male_result.html", result=condition)

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)

