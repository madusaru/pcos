import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# -------------------------
# CREATE MODEL FOLDER
# -------------------------
os.makedirs("model", exist_ok=True)

# -------------------------
# PATHS
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR   = os.path.join(BASE_DIR, "dataset", "val")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "test")

MODEL_PATH = os.path.join(BASE_DIR, "model", "pcos_mobilenet_model.h5")

IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 20

print("📁 Train Dir:", TRAIN_DIR)
print("📁 Val Dir:", VAL_DIR)
print("📁 Test Dir:", TEST_DIR)
print("💾 Model will be saved at:", MODEL_PATH)

# -------------------------
# DATA GENERATORS
# -------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

# -------------------------
# MODEL (EfficientNet)
# -------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("✅ Model built successfully")
print(model.summary())

# -------------------------
# TRAIN PHASE 1
# -------------------------
print("\n🚀 Starting Training (Phase 1)...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# -------------------------
# FINE-TUNING
# -------------------------
print("\n🔧 Fine-tuning model...")

base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# -------------------------
# SAVE MODEL
# -------------------------
model.save(MODEL_PATH)
print(f"\n✅ Model saved at: {MODEL_PATH}")

# -------------------------
# TEST EVALUATION
# -------------------------
print("\n📊 Evaluating on test data...")
test_loss, test_acc = model.evaluate(test_gen)

print(f"\n🎯 Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

