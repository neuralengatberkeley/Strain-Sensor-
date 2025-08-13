import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

# === Parameters ===
frame_size = (1200, 1920)  # height, width
bayer_pattern = cv2.COLOR_BayerRG2RGB  # use cv2.COLOR_BayerRG2GRAY for grayscale

# === Select parent folder ===
root = tk.Tk()
root.withdraw()
parent_dir = filedialog.askdirectory(title="Select parent folder with cam_top and cam_side")

if not parent_dir:
    print("No folder selected.")
    exit()

print(f"Selected folder: {parent_dir}")

cam_folders = ['cam_top', 'cam_side']

for cam_name in cam_folders:
    folder = os.path.join(parent_dir, cam_name)
    if not os.path.isdir(folder):
        print(f"Skipping missing folder: {folder}")
        continue

    raw_files = [f for f in os.listdir(folder) if f.endswith('.raw')]
    if not raw_files:
        print(f"No .raw files found in {folder}")
        continue

    print(f"Processing {len(raw_files)} files in {folder}")

    # Create subfolder for JPEGs
    jpeg_folder = os.path.join(folder, "jpg")
    os.makedirs(jpeg_folder, exist_ok=True)

    for raw_file in raw_files:
        raw_path = os.path.join(folder, raw_file)
        jpg_filename = os.path.splitext(raw_file)[0] + '.jpg'
        jpg_path = os.path.join(jpeg_folder, jpg_filename)

        # Read raw binary data
        with open(raw_path, 'rb') as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8)

        try:
            bayer_img = raw_data.reshape(frame_size)
        except ValueError:
            print(f"Skipping {raw_file}: incorrect size.")
            continue

        # Convert Bayer to RGB
        rgb_img = cv2.cvtColor(bayer_img, bayer_pattern)

        # Save as JPEG
        cv2.imwrite(jpg_path, rgb_img)

    print(f"Finished processing {cam_name}")

print("All done.")


