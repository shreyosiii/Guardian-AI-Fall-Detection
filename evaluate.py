import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURATION (MUST MATCH TRAINING) ---
IMG_SIZE        = 64
SEQUENCE_LENGTH = 10
DATASET_DIR     = "frames"

def load_evaluation_data(folder):
    sequences = []
    labels    = []
    categories = ["fall", "no_fall"]

    print("Preparing data for evaluation...")

    for label, category in enumerate(categories):
        path = os.path.join(folder, category)
        if not os.path.exists(path):
            print(f"[WARNING] Folder not found: {path}")
            continue

        file_list = os.listdir(path)
        try:
            frames = sorted(file_list, key=lambda x: int(''.join(filter(str.isdigit, x))))
        except:
            frames = sorted(file_list)

        print(f"Loading {category} ({len(frames)} images)...")

        temp = []
        for img_name in frames:
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(path, img_name)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            temp.append(frame)

            if len(temp) == SEQUENCE_LENGTH:
                sequences.append(temp)
                labels.append(label)
                temp = []

    return np.array(sequences), np.array(labels)


# ── 1. Load Data ──────────────────────────────────────────────────────────────
X_test, y_true = load_evaluation_data(DATASET_DIR)
print(f"Total sequences to test: {len(X_test)}")

if len(X_test) == 0:
    print("❌ No data found. Check your frames/ folder structure.")
    exit()

# ── 2. Load Model ─────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model("fall_video_model.h5")
print("✅ Model loaded.")

# ── 3. Predict ────────────────────────────────────────────────────────────────
print("Running predictions...")
y_pred_prob = model.predict(X_test, verbose=1)

# Fixed: pred < 0.5 = Fall (class 0), pred > 0.5 = No Fall (class 1)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# ── 4. Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 40)
print("       FINAL EVALUATION REPORT")
print("=" * 40)

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred,
      target_names=["Fall (Class 0)", "No Fall (Class 1)"]))

cm = confusion_matrix(y_true, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)
print(f"\n  Correct Falls Detected : {cm[0][0]}")
print(f"  Missed Falls (Danger!) : {cm[0][1]}")
print(f"  False Alarms           : {cm[1][0]}")
print(f"  Correct Normal         : {cm[1][1]}")
print("=" * 40)

# ── 5. Plot Confusion Matrix ──────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=["Predicted Fall", "Predicted Normal"],
            yticklabels=["Actual Fall", "Actual Normal"])
plt.title(f"Confusion Matrix  |  Accuracy: {acc*100:.2f}%", fontsize=14)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=150)
plt.show()
print("\n📊 Confusion matrix saved as: evaluation_results.png")