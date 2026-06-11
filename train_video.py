import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE        = 64
SEQUENCE_LENGTH = 10
DATASET_DIR     = "frames"
EPOCHS          = 30          # Reduced from 50 — early stopping handles the rest
BATCH_SIZE      = 8           # Increased from 4 for stability

# ── MEMORY LIMIT ──────────────────────────────────────────────────────────────
# 8GB RAM Mac — use max 1500 images per category to stay safe
MAX_IMAGES_PER_CLASS = 1500

def load_sequences(folder):
    sequences = []
    labels    = []
    categories = ["fall", "no_fall"]
    print(f"MAPPING: {categories[0]} = 0, {categories[1]} = 1")

    for label, category in enumerate(categories):
        path = os.path.join(folder, category)
        if not os.path.exists(path):
            print(f"[WARNING] Folder not found: {path}")
            continue

        file_list = os.listdir(path)

        # Filter only image files
        file_list = [f for f in file_list
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Sort frames
        try:
            frames = sorted(file_list,
                            key=lambda x: int(''.join(filter(str.isdigit, x))))
        except:
            frames = sorted(file_list)

        # ── MEMORY LIMIT: cap at MAX_IMAGES_PER_CLASS ─────────────────────
        if len(frames) > MAX_IMAGES_PER_CLASS:
            print(f"  [{category}] Found {len(frames)} images — "
                  f"limiting to {MAX_IMAGES_PER_CLASS} to save RAM")
            # Take evenly spaced samples so we cover the whole dataset
            step   = len(frames) // MAX_IMAGES_PER_CLASS
            frames = frames[::step][:MAX_IMAGES_PER_CLASS]
        else:
            print(f"  [{category}] Found {len(frames)} images")

        temp = []
        seq_count = 0
        print(f"  Building sequences for {category}...")

        for img_name in frames:
            img_path = os.path.join(path, img_name)
            frame    = cv2.imread(img_path)
            if frame is None:
                continue

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            temp.append(frame)

            if len(temp) == SEQUENCE_LENGTH:
                sequences.append(np.array(temp, dtype=np.float32))
                labels.append(label)
                temp = []
                seq_count += 1

        print(f"  [{category}] Created {seq_count} sequences")

    return np.array(sequences, dtype=np.float32), np.array(labels)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("=" * 50)
print("  Fall Detection — Training Script")
print("=" * 50)
print(f"\nLoading sequences (max {MAX_IMAGES_PER_CLASS} images per class)...")

X, y = load_sequences(DATASET_DIR)

if len(X) == 0:
    print("[ERROR] No data loaded. Check your frames/ folder.")
    exit()

X, y = shuffle(X, y, random_state=42)

print(f"\nData shape : {X.shape}")
print(f"Labels     : {np.sum(y==0)} fall, {np.sum(y==1)} no_fall")
print(f"Memory used: ~{X.nbytes / 1e6:.1f} MB")

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
print("\nBuilding CNN-LSTM model...")

model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu'),
                    input_shape=(SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3)),
    TimeDistributed(MaxPooling2D()),
    TimeDistributed(Flatten()),
    Dropout(0.5),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'fall_video_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print(f"\nStarting training — {EPOCHS} epochs max (early stopping enabled)...")
history = model.fit(
    X, y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# ── RESULTS ───────────────────────────────────────────────────────────────────
best_acc = max(history.history['val_accuracy'])
print(f"\n{'='*50}")
print(f"  Training Complete!")
print(f"  Best Validation Accuracy: {best_acc*100:.2f}%")
print(f"  Model saved: fall_video_model.h5")
print(f"{'='*50}")
