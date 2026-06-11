import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SEQUENCE_LENGTH  = 10
IMG_SIZE         = 64
STEP_SIZE        = 5
FALL_THRESHOLD   = 0.95   # Below this = FALL
SAVE_OUTPUT      = True

print("Loading model...")
try:
    model = load_model("fall_video_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


def predict_video(video_path):
    print(f"\nProcessing: {video_path}")

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print("Available videos in videos folder:")
        # Auto find videos
        for root, dirs, files in os.walk("videos"):
            for f in files:
                if f.endswith('.mp4'):
                    print(f"  {os.path.join(root, f)}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    # Save output video
    writer = None
    if SAVE_OUTPUT:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = video_path.replace(".mp4", "_result.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height)
        )
        print(f"Saving result to: {out_path}")

    frames_buffer      = []
    fall_detected_flag = False
    frame_count        = 0
    fall_count         = 0
    last_pred          = 1.0
    status             = "Loading..."
    color              = (255, 255, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Preprocess
        resized    = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        frames_buffer.append(normalized)

        if len(frames_buffer) > SEQUENCE_LENGTH:
            frames_buffer.pop(0)

        # Predict every STEP_SIZE frames
        if len(frames_buffer) == SEQUENCE_LENGTH and frame_count % STEP_SIZE == 0:
            input_data = np.array(frames_buffer).reshape(
                1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 3
            )
            last_pred = float(model.predict(input_data, verbose=0)[0][0])

            if last_pred < FALL_THRESHOLD:
                fall_count += 1
                fall_detected_flag = True
                status = "FALL DETECTED"
                color  = (0, 0, 255)
                print(f"Frame {frame_count}: FALL (score={last_pred:.4f})")
            else:
                status = "Normal"
                color  = (0, 220, 80)
                print(f"Frame {frame_count}: Normal (score={last_pred:.4f})")

        # Draw HUD
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(frame, status, (15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame,
                    f"Frame: {frame_count}  Score: {last_pred:.3f}",
                    (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Red border on fall
        if status == "FALL DETECTED":
            cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 5)

        cv2.imshow("Fall Detection - Video Test", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit by user.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("\n" + "-" * 40)
    print(f"Total frames processed : {frame_count}")
    print(f"Fall events detected   : {fall_count}")
    if fall_detected_flag:
        print("FINAL RESULT: FALL DETECTED")
    else:
        print("FINAL RESULT: No Fall Detected")
    print("-" * 40)


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Auto detect OS and use correct path separator
    if sys.platform == "win32":
        # Windows paths
        fall_video   = os.path.join("videos", "fall",    "Raw_Video", "20240912_102330.mp4")
        normal_video = os.path.join("videos", "No_Fall", "Raw_Video", "C_N_450.mp4")
    else:
        # Mac/Linux paths
        fall_video   = "videos/fall/Raw_Video/20240912_102330.mp4"
        normal_video = "videos/No_Fall/Raw_Video/C_N_450.mp4"

    print("=" * 40)
    print("Which video do you want to test?")
    print("1. Fall video")
    print("2. Normal video")
    print("3. Enter custom path")
    print("=" * 40)

    choice = input("Enter 1, 2 or 3: ").strip()

    if choice == "1":
        predict_video(fall_video)
    elif choice == "2":
        predict_video(normal_video)
    elif choice == "3":
        custom = input("Enter video path: ").strip()
        predict_video(custom)
    else:
        print("Invalid choice. Running fall video by default.")
        predict_video(fall_video)