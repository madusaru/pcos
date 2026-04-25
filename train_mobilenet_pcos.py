import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ================= SETTINGS =================
IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 15

# ================= DATA GENERATORS =================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True
).flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ================= SHOW CLASS ORDER =================
print("\nClass label mapping:")
print(train_gen.class_indices)
print("\nREMEMBER THIS ORDER for prediction code\n")

# ================= LOAD PRETRAINED MODEL =================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
for layer in base_model.layers:
    layer.trainable = False

# ================= CUSTOM CLASSIFIER =================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ================= COMPILE =================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ================= SAVE MODEL =================
model.save("pcos_mobilenet_model.h5")

print("\n✅ Training finished")
print("✅ Model saved as pcos_mobilenet_model.h5")
