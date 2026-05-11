# 🌸 AI-Based Fertility Prediction System

> An intelligent web application that analyzes medical reports to predict fertility status for both male and female users using Machine Learning and OCR.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=flat-square&logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Infertility affects millions of individuals worldwide, yet timely diagnosis remains a challenge due to limited medical resources and time-consuming manual analysis. This system automates fertility assessment by:

- Detecting **PCOS risk** in female users based on clinical parameters
- Evaluating **sperm quality** for male users via uploaded reports
- Using **OCR** to extract data directly from medical report images
- Presenting results in **plain, non-technical language**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧬 Female Fertility Analysis | PCOS risk prediction using hormonal and cycle parameters |
| 🔬 Male Fertility Analysis | Sperm count, motility, and morphology evaluation |
| 📄 OCR Report Extraction | Upload JPG/PNG/PDF lab reports; values extracted automatically |
| 🤖 ML-Powered Predictions | Random Forest Classifier with ~99% accuracy on test data |
| 🖥️ Web Interface | Simple, accessible Flask-based front-end |
| 💬 Plain-Language Results | Technical outputs translated into user-friendly explanations |

---

## 🏗️ System Architecture

```
User Selects Gender
      │
      ├──── Female User
      │         │
      │         ├── Enter PCOS Inputs (Weight, Height, Cycle, Symptoms)
      │         │         │
      │         │    PCOS Prediction Model
      │         │         │
      │         │    High Confidence? ──► Upload Fertility Report Image
      │         │                                   │
      │         │                        OCR Text Extraction (Hormones)
      │         │                                   │
      │         │                       Female Fertility Prediction
      │
      └──── Male User
                │
          Upload Sperm Report Image
                │
          OCR Text Extraction (Sperm Parameters)
                │
          Male Fertility Evaluation
```

---

## 🧩 Modules

1. **User Input & Profile** — Collects gender-specific health parameters and validates inputs
2. **PCOS Prediction** — ML model estimates PCOS likelihood from entered health data
3. **Report Upload & Image Processing** — Handles image uploads with preprocessing (noise reduction, clarity enhancement)
4. **OCR Data Extraction** — Pytesseract extracts clinical values; regex isolates key parameters
5. **Fertility Prediction** — Trained models assess fertility status for both male and female users
6. **Result Simplification** — Converts medical terminology into accessible, friendly language
7. **Output Display** — Web dashboard shows fertility status, PCOS risk level, and explanations

---

## 🛠️ Tech Stack

**Backend**
- Python 3.8+
- Flask
- Scikit-learn (Random Forest Classifier)
- TensorFlow / Keras
- Pytesseract (OCR)
- Pillow (image processing)
- Pandas / NumPy

**Frontend**
- HTML5 & CSS3

**Tools**
- Joblib (model serialization)
- Regex (parameter extraction)

---

## 📊 Dataset & Model

- **Dataset**: Synthetic dataset of **1,000 samples** generated to simulate real-world fertility conditions
- **Features used**:
  - Age, AMH, FSH, LH, Progesterone
  - Follicle Count, Ovarian Volume
  - Cycle Length, Cycle Regularity
- **Labeling**: Clinical rule-based logic (Fertile / Infertile)
- **Split**: 80% training / 20% testing
- **Model**: Random Forest Classifier
- **Accuracy**: ~99% on test set

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install flask scikit-learn tensorflow keras pytesseract pillow pandas numpy joblib
```

> Also install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) and update the path in the code.

### Installation

```bash
git clone https://github.com/madusaru/pcos.git
cd pcos
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`.

---

## 🔍 Sample Code — OCR-Based Fertility Prediction

```python
# predict_from_image.py
import pytesseract
from PIL import Image
import re
import pandas as pd
import joblib

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load trained model
model = joblib.load("fertility_model.pkl")

# Load and process report image
img = Image.open("lab_report.png")
text = pytesseract.image_to_string(img)

# Extract hormone values using regex
def extract_value(pattern, text):
    match = re.search(pattern, text, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0

amh = extract_value(r"AMH\s*[:=]?\s*([0-9.]+)", text)
fsh = extract_value(r"FSH\s*[:=]?\s*([0-9.]+)", text)
lh  = extract_value(r"LH\s*[:=]?\s*([0-9.]+)", text)

# Predict
user_df = pd.DataFrame([{"AMH": amh, "FSH": fsh, "LH": lh}])
prediction = model.predict(user_df)[0]
print("Predicted Fertility Status:", prediction)
```

## 🔮 Future Enhancements

- Support for PDF report uploads
- Expanded hormone panel and additional clinical markers
- Integration of deep learning for improved image-based analysis
- Multi-language support for broader accessibility
- Doctor referral suggestions based on prediction confidence
- Mobile-responsive UI

---

## 👩‍💻 Team

| SRN | Name | Contact |
|-----|------|---------|
| R23EA101 | S Madumitha | madusaravana21@gmail.com |
| R23EA103 | Monisha S | Shivmonisha77@gmail.com |

**Institution**: School of Computing Science and Engineering, REVA University, Bengaluru  
**Academic Year**: 2025–26 | Semester 6  
**Course**: Mini Project – Research Based (B22CI0605 / B22EH0605)

---

## 📚 References

Key references informing this project:

- Agirsoy et al. (2025) — Non-invasive PCOS diagnosis using ML from ultrasound and clinical features
- Barua et al. (2026) — AI-powered sperm analysis via deep learning
- Zheng et al. (2023) — Extracting laboratory test information from paper-based reports
- Li et al. (2024) — Tabular data extraction from scanned lab reports using deep learning
- Adinugroho & Nakazawa (2024) — Deep learning-based sperm motility and morphology estimation

---

## ⚠️ Disclaimer

This system is developed as an academic mini-project and is **not a substitute for professional medical advice**. Predictions are based on extracted parameters and trained models; users should always consult a qualified healthcare provider for medical decisions.

