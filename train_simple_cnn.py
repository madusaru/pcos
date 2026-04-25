import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 25

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
).flow_from_directory(
    "dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dataset/val",
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary"
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

model.save("pcos_simple_cnn.h5")
