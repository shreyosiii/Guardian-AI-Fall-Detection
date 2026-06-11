import os
import sys
import subprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- CONFIGURATION ---
# This determines how many new copies to make for every 1 original image.
AUGMENTATION_FACTOR = 5

# Use script-relative paths so behavior is consistent when run from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "frames")  # The folder containing 'fall' and 'no_fall'
IMG_SIZE = 128

# Define how to "remix" the images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


def try_extract_frames():
    """If `frames` is missing but `videos` exists, run extract_frames.py to create frames."""
    videos_dir = os.path.join(BASE_DIR, "videos")
    if os.path.exists(videos_dir) and any(os.scandir(videos_dir)):
        print("Frames folder missing — running extract_frames.py to populate frames from videos...")
        cmd = [sys.executable, os.path.join(BASE_DIR, "extract_frames.py")]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"extract_frames.py failed: {e}")
    else:
        # Create empty structure so augmenter can run without crashing
        os.makedirs(os.path.join(DATASET_DIR, "fall"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "no_fall"), exist_ok=True)


def augment_category(category):
    folder_path = os.path.join(DATASET_DIR, category)

    # Ensure folder exists (try to extract frames if possible)
    if not os.path.exists(folder_path):
        try_extract_frames()

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found and could not be created.\nRun extract_frames.py first.")
        return

    # Get list of images
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No images found in '{category}' — nothing to augment.")
        return

    print(f"Found {len(images)} original images in '{category}'. Generating new data...")

    total_new = 0

    for image_name in images:
        # Skip images that were already created by this script (start with 'aug_')
        if image_name.startswith("aug_"):
            continue

        try:
            # Load and resize the image to ensure consistent shapes
            img_path = os.path.join(folder_path, image_name)
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

            # Generate new versions. Use prefix 'aug_' to match the skip check above.
            i = 0
            for _ in datagen.flow(x, batch_size=1,
                                  save_to_dir=folder_path,
                                  save_prefix='aug_',
                                  save_format='jpg'):
                i += 1
                total_new += 1
                if i >= AUGMENTATION_FACTOR:
                    break  # Stop after creating the desired number of copies
        except Exception as e:
            print(f"Skipping {image_name}: {e}")

    print(f"✅ Finished! Created {total_new} new images for '{category}'.")


# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    print("Starting Data Augmentation...")
    augment_category("fall")
    augment_category("no_fall")
    print("\nData Augmentation Complete. You can now run train_video.py.")