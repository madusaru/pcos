# evaluate_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "best_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)


# -----------------------------
# Test data generator
# -----------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

print("\nClass mapping:", test_generator.class_indices)

# -----------------------------
# Predictions
# -----------------------------
y_true = test_generator.classes
y_probs = model.predict(test_generator)
y_pred = (y_probs > 0.5).astype(int).reshape(-1)

# -----------------------------
# Classification Report
# -----------------------------
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# ROC–AUC Curve
# -----------------------------
auc = roc_auc_score(y_true, y_probs)
fpr, tpr, _ = roc_curve(y_true, y_probs)

print(f"\n🔥 ROC–AUC Score: {auc:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
