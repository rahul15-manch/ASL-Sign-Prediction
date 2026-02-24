# src/train.py
import os
import sys
import json
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kagglehub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models import create_cnn_model

# ---------------------------
# Download / locate dataset
# ---------------------------
print("[INFO] Resolving dataset path via kagglehub...")
DATASET_PATH = kagglehub.dataset_download("grassknoted/asl-alphabet")
print(f"[INFO] Dataset path: {DATASET_PATH}")

# The dataset has a subfolder: asl_alphabet_train/asl_alphabet_train/
TRAIN_SUBDIR = os.path.join(DATASET_PATH, "asl_alphabet_train", "asl_alphabet_train")
if not os.path.isdir(TRAIN_SUBDIR):
    # Fallback: try top-level
    TRAIN_SUBDIR = os.path.join(DATASET_PATH, "asl_alphabet_train")
DATA_DIR = TRAIN_SUBDIR
print(f"[INFO] Using training data from: {DATA_DIR}")

# ---------------------------
# Paths & Hyperparameters
# ---------------------------
MODEL_PATH   = "models/sign_model.h5"
BACKUP_PATH  = "models/sign_model_backup.h5"
PLOT_PATH    = "models/training_history.png"
IMG_SIZE     = (64, 64)
BATCH_SIZE   = 32
EPOCHS       = 30   # EarlyStopping will stop early if needed

# ---------------------------
# Backup existing model
# ---------------------------
if os.path.exists(MODEL_PATH):
    shutil.copy(MODEL_PATH, BACKUP_PATH)
    print(f"[INFO] Existing model backed up to {BACKUP_PATH}")

# ---------------------------
# Data Generators  ← KEY FIX
# Separate datagens so augmentation is NOT applied to validation data
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Validation: rescale ONLY — no augmentation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"[INFO] Training samples : {train_generator.samples}")
print(f"[INFO] Validation samples: {val_generator.samples}")
print(f"[INFO] Classes: {train_generator.num_classes}")

# ---------------------------
# Model
# ---------------------------
model = create_cnn_model(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    num_classes=train_generator.num_classes
)
model.summary()

# ---------------------------
# Callbacks  ← NEW
# ---------------------------
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ---------------------------
# Train
# ---------------------------
print("[INFO] Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------------------
# Save Model
# ---------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")

# ---------------------------
# Save Class Mapping
# ---------------------------
class_indices = train_generator.class_indices
with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f)
print("[INFO] Class mapping saved to models/class_indices.json")

# ---------------------------
# Save Training Plots
# ---------------------------
acc     = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss    = history.history.get("loss", [])
val_loss= history.history.get("val_loss", [])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(acc,     label="Train Accuracy")
plt.plot(val_acc, label="Val Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.title("Accuracy — Train vs Validation")
plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss,     label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss — Train vs Validation")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"[INFO] Training history plot saved to {PLOT_PATH}")
