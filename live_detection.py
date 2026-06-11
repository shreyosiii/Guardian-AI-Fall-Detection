import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from tensorflow.keras.models import load_model

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SEQUENCE_LENGTH      = 10
IMG_SIZE             = 64
PREDICT_EVERY        = 5
FALL_THRESHOLD       = 0.95   # score < 0.95 = FALL
REQUIRED_CONSECUTIVE = 2
ALERT_COOLDOWN       = 8      # seconds between alerts

# ── SNAPSHOT FOLDER ───────────────────────────────────────────────────────────
SNAPSHOT_DIR = "fall_evidence"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ── VOICE ALERT ───────────────────────────────────────────────────────────────
def play_alert():
    """Works on Windows, Mac and Linux"""
    try:
        if sys.platform == "win32":
            # Windows voice alert
            import subprocess
            subprocess.call([
                "PowerShell", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 1; "
                "$s.Speak('Alert! A person has fallen down! Immediate assistance required!')"
            ])
        elif sys.platform == "darwin":
            # Mac voice alert
            import subprocess
            subprocess.call(["say", "-v", "Samantha",
                           "Alert! A person has fallen down! Immediate assistance required!"])
        else:
            # Linux
            import subprocess
            subprocess.call(["espeak",
                           "Alert! A person has fallen down! Immediate assistance required!"])
    except Exception as e:
        print(f"[ALERT] Voice alert error: {e}")
        # Fallback — print alert if voice fails
        print("\n" + "!"*50)
        print("!!! ALERT: FALL DETECTED !!!")
        print("!"*50 + "\n")

def trigger_alert():
    """Run alert in background so video doesn't freeze"""
    t = threading.Thread(target=play_alert, daemon=True)
    t.start()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("=" * 50)
print("  Fall Detection - Live Detection")
print("=" * 50)
print("\nLoading model...")

try:
    model = load_model("fall_video_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure fall_video_model.h5 is in the Fall folder")
    exit()

# ── OPEN WEBCAM ───────────────────────────────────────────────────────────────
print("Opening webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open webcam!")
    print("Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("Webcam opened successfully!")
print("Press Q to quit\n")

# Test alert on startup
print("Testing voice alert...")
trigger_alert()
time.sleep(2)

# ── VARIABLES ─────────────────────────────────────────────────────────────────
frames_buffer   = []
fall_counter    = 0
frame_count     = 0
last_pred       = 1.0
last_alert_time = 0
total_falls     = 0
status          = "Loading buffer..."
color           = (255, 255, 0)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from webcam!")
        break

    frame_count += 1
    display = cv2.flip(frame, 1)  # Mirror effect

    # ── Preprocess ────────────────────────────────────────────────────────
    resized    = cv2.resize(display, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    frames_buffer.append(normalized)

    if len(frames_buffer) > SEQUENCE_LENGTH:
        frames_buffer.pop(0)

    # ── Predict ───────────────────────────────────────────────────────────
    if len(frames_buffer) == SEQUENCE_LENGTH and frame_count % PREDICT_EVERY == 0:
        inp       = np.array(frames_buffer).reshape(
            1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3
        )
        last_pred = float(model.predict(inp, verbose=0)[0][0])

        if last_pred < FALL_THRESHOLD:
            fall_counter += 1
        else:
            fall_counter = 0
            status = "Normal"
            color  = (0, 220, 80)

        # Trigger alert
        if fall_counter >= REQUIRED_CONSECUTIVE:
            status = "FALL DETECTED"
            color  = (0, 0, 255)

            now = time.time()
            if now - last_alert_time > ALERT_COOLDOWN:
                total_falls    += 1
                last_alert_time = now

                # Save snapshot
                ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(SNAPSHOT_DIR, f"fall_{ts}.jpg")
                cv2.imwrite(snap_path, display)
                print(f"Fall #{total_falls} detected! Snapshot: {snap_path}")

                # Voice alert
                trigger_alert()

    # ── Draw HUD ──────────────────────────────────────────────────────────
    h, w = display.shape[:2]

    # Top dark bar
    cv2.rectangle(display, (0, 0), (w, 70), (15, 15, 15), -1)

    # Status text
    cv2.putText(display, status, (15, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    # Fall count top right
    cv2.putText(display, f"Falls: {total_falls}",
                (w - 150, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)

    # Fall risk bar
    bar_w = int((1.0 - last_pred) * (w - 40))
    bar_c = (0, 0, 255) if last_pred < FALL_THRESHOLD else (0, 200, 80)
    cv2.rectangle(display, (20, 73), (20 + bar_w, 86), bar_c, -1)
    cv2.rectangle(display, (20, 73), (w - 20, 86), (80, 80, 80), 1)
    cv2.putText(display, "Fall Risk", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Score bottom
    cv2.putText(display,
                f"Score: {last_pred:.3f}  |  Fall Risk: {(1-last_pred):.3f}",
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    # Buffer loading indicator
    if len(frames_buffer) < SEQUENCE_LENGTH:
        cv2.putText(display,
                    f"Loading buffer: {len(frames_buffer)}/{SEQUENCE_LENGTH}",
                    (15, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    # Flashing red border on fall
    if fall_counter >= REQUIRED_CONSECUTIVE:
        thickness = 8 if (frame_count // 5) % 2 == 0 else 3
        cv2.rectangle(display, (0, 0), (w-1, h-1), (0, 0, 255), thickness)

    # Show window
    cv2.imshow("Fall Detection - Live", display)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit by user.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print(f"\nSession ended. Total falls detected: {total_falls}")