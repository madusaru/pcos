import os
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import OrderedDict

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 123
DATA_DIR = "dataset"            # expects dataset/train/CLASS and dataset/test/CLASS
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_SPLIT = 0.2                 # 20% of train used as validation

# ---------------------------
# Quick check: paths exist
# ---------------------------
if not os.path.isdir(TRAIN_DIR):
    raise SystemExit(f"Train folder not found: {TRAIN_DIR}")
if not os.path.isdir(TEST_DIR):
    print(f"Warning: test folder not found at {TEST_DIR}. You can still train (no separate test).")

# ---------------------------
# Create train/val/test datasets using validation_split
# ---------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="training",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VAL_SPLIT,
    subset="validation",
)

# optional test dataset if you have it
if os.path.isdir(TEST_DIR):
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
else:
    test_ds = None

# ---------------------------
# Count images per class (for class weights)
# ---------------------------
class_names = train_ds.class_names  # consistent sorted order
train_counts = OrderedDict()
for cls in class_names:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    # count files only (common image extensions)
    cnt = sum(1 for f in os.listdir(cls_dir) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp')))
    train_counts[cls] = cnt

print("Training counts by class:", train_counts)

total = sum(train_counts.values())
n_classes = len(train_counts)
class_weight = {}
for i, cls in enumerate(class_names):
    # balanced weight formula: total/(n_classes * count)
    class_weight[i] = total / (n_classes * train_counts[cls])
print("Using class_weight:", class_weight)

# ---------------------------
# Performance: prefetch / cache
# ---------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
if test_ds is not None:
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------
# Data augmentation + normalization
# ---------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05),
])

normalization = layers.Rescaling(1./255)

# ---------------------------
# Build model (transfer learning)
# ---------------------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    weights='imagenet'
)
base_model.trainable = False  # freeze for initial training

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = normalization(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

model.summary()

# ---------------------------
# Callbacks
# ---------------------------
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("pcos_model.h5", save_best_only=True, monitor="val_loss")
earlystopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# ---------------------------
# Train
# ---------------------------
EPOCHS = 15
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystopping_cb],
    class_weight=class_weight
)

# ---------------------------
# Optional: unfreeze and fine-tune
# ---------------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

fine_tune_epochs = 10
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs,
    callbacks=[checkpoint_cb, earlystopping_cb],
    class_weight=class_weight
)

# ---------------------------
# Evaluate on test if available
# ---------------------------
if test_ds is not None:
    print("Evaluating on test set:")
    results = model.evaluate(test_ds)
    print("Test results (loss, accuracy, precision, recall):", results)
