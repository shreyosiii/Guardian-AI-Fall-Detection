import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(video_path):
        print(f"Skipping: {video_path} (File not found)")
        return

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/{video_name}_frame_{count}.jpg", frame)
        count += 1

    cap.release()
    print(f"Done: {video_name} ({count} frames extracted)")


# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

# Fixed: matches your actual folder structure
video_folders = {
    "fall":    os.path.join(VIDEOS_DIR, "fall",    "Raw_Video"),
    "no_fall": os.path.join(VIDEOS_DIR, "No_Fall", "Raw_Video"),
}

for cat, video_input_dir in video_folders.items():
    frame_output_dir = os.path.join(FRAMES_DIR, cat)
    os.makedirs(frame_output_dir, exist_ok=True)

    if not os.path.exists(video_input_dir):
        print(f"Error: {video_input_dir} not found!")
        continue

    print(f"\nProcessing category: {cat}")
    print(f"Reading from : {video_input_dir}")
    print(f"Saving to    : {frame_output_dir}")

    for video_file in os.listdir(video_input_dir):
        if video_file.lower().endswith(('.mp4', '.avi', '.mov')):
            full_path = os.path.join(video_input_dir, video_file)
            extract_frames(full_path, frame_output_dir)

print("\nAll done! Check your frames/ folder.")