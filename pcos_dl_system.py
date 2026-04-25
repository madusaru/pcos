# ---------------------------------------
# PCOS COMPLETE SYSTEM (DL + Fine Tuning + Augmentation)
# ---------------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# -----------------------------
# PATHS & SETTINGS
# -----------------------------
DATASET_DIR = "dataset/train"
VAL_DIR = "dataset/val"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10
THRESHOLD = 0.7
MODEL_PATH = "pcos_cnn_model.h5"

# -----------------------------
# SYMPTOM CHECK
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
        print("\n✅ LOW PCOS RISK (No need for image test)")
        return False

# -----------------------------
# CLEAN DATASET
# -----------------------------
def clean_dataset(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                os.remove(path)
                continue
            try:
                image.load_img(path, target_size=IMG_SIZE)
            except Exception:
                print(f"Removed corrupted image: {path}")
                os.remove(path)

# -----------------------------
# CREATE MODEL (with Data Augmentation)
# -----------------------------
def create_model():
    # Data augmentation layers to reduce overfitting
    train_augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2)
    ])

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(224,224,3))
    x = train_augment(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model():
    clean_dataset(DATASET_DIR)
    clean_dataset(VAL_DIR)

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(DATASET_DIR,
                                                  target_size=IMG_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='binary')
    val_gen = val_datagen.flow_from_directory(VAL_DIR,
                                              target_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE,
                                              class_mode='binary')

    model = create_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ]

    # Initial Training (Feature Extraction)
    model.fit(train_gen,
              validation_data=val_gen,
              epochs=INITIAL_EPOCHS,
              callbacks=callbacks)

    # Fine Tune: Unfreeze some layers
    base_model = model.layers[2]  # This is EfficientNet backbone
    base_model.trainable = True

    # Fine-tuning from a small learning rate
    model.compile(optimizer=Adam(1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    model.fit(train_gen,
              validation_data=val_gen,
              epochs=FINE_TUNE_EPOCHS,
              callbacks=callbacks)

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved at {MODEL_PATH}")
    return model

# -----------------------------
# IMAGE PREDICTION
# -----------------------------
def predict_image(model, img_path):
    if not os.path.exists(img_path):
        print("❌ Image file not found!")
        return

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_prob = model.predict(img_array)[0][0]
    print("\n--- IMAGE RESULT ---")
    print(f"PCOS Probability: {pred_prob:.2f}")

    if pred_prob > THRESHOLD:
        print(f"🩺 FINAL RESULT: PCOS DETECTED (threshold {THRESHOLD})")
    else:
        print(f"✅ FINAL RESULT: NON-PCOS (threshold {THRESHOLD})")

# -----------------------------
# MAIN FLOW
# -----------------------------
if __name__ == "__main__":
    high_risk = symptom_check()

    if high_risk:
        if os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            model = tf.keras.models.load_model(MODEL_PATH)
        else:
            print("Training new CNN model...")
            model = train_model()

        img_path = input("\nEnter ultrasound image path: ")
        predict_image(model, img_path)
