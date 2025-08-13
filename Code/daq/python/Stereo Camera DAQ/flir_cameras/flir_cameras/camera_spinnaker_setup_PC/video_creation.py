import os
import cv2
import tkinter as tk
from tkinter import filedialog
from natsort import natsorted

# === Helper function ===
def get_sorted_images(jpg_folder):
    images = [f for f in os.listdir(jpg_folder) if f.lower().endswith('.jpg')]
    return natsorted(images)  # natural sort

# === GUI dialogs ===
root = tk.Tk()
root.withdraw()

# Step 1: Select parent folder containing timestamped subfolders
source_root = filedialog.askdirectory(title="Select folder with timestamped subfolders (each contains cam_top and cam_side)")
if not source_root:
    print("No source folder selected. Exiting.")
    exit()

# Step 2: Select folder to save videos
save_root = filedialog.askdirectory(title="Select folder to save MP4 videos")
if not save_root:
    print("No save folder selected. Exiting.")
    exit()

# === Process each timestamped subfolder ===
subfolders = natsorted([f for f in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, f))])
video_counter = 0

for subfolder in subfolders:
    subfolder_path = os.path.join(source_root, subfolder)
    cam_folders = {
        "camera - 1": os.path.join(subfolder_path, "cam_side", "jpg"),
        "camera - 2": os.path.join(subfolder_path, "cam_top", "jpg")
    }

    for camera_label, jpg_path in cam_folders.items():
        if not os.path.isdir(jpg_path):
            print(f"Skipping missing path: {jpg_path}")
            continue

        images = get_sorted_images(jpg_path)
        if not images:
            print(f"No JPEG images found in {jpg_path}")
            continue

        # Read first image to determine frame size
        first_image_path = os.path.join(jpg_path, images[0])
        frame = cv2.imread(first_image_path)
        if frame is None:
            print(f"Failed to read first image: {first_image_path}")
            continue

        height, width, _ = frame.shape

        # Create subdirectory for camera label if it doesn't exist
        camera_save_dir = os.path.join(save_root, camera_label)
        os.makedirs(camera_save_dir, exist_ok=True)

        # Use original naming convention but save into the appropriate subfolder
        out_name = f"vid {video_counter} {camera_label}.mp4"
        out_path = os.path.join(camera_save_dir, out_name)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30  # Adjust as needed
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print(f"Writing {out_name} with {len(images)} frames...")

        for img_name in images:
            img_path = os.path.join(jpg_path, img_name)
            frame = cv2.imread(img_path)
            if frame is not None:
                out.write(frame)

        out.release()
        print(f"Saved: {out_path}")

    video_counter += 1

print("All videos created.")



