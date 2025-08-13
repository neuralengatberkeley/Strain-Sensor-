import os
import ffmpeg

# === Patch PATH so ffmpeg-python can find ffmpeg.exe ===
ffmpeg_bin_path = r"C:\Users\toppe\OneDrive\Desktop\ffmpeg-2025-07-23-git-829680f96a-essentials_build\bin"
os.environ["PATH"] = ffmpeg_bin_path + ";" + os.environ["PATH"]

# === Reencoding Script ===
base_dir = r"C:\flir_capture\videos"
folders = ["camera - 1", "camera - 2"]
valid_exts = [".mp4", ".avi", ".mov", ".mkv"]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    print(f"Processing folder: {folder_path}")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_root, ext = os.path.splitext(filename)

        if ext.lower() not in valid_exts:
            continue

        output_path = os.path.join(folder_path, f"{file_root}_reencoded{ext}")
        print(f"Reencoding: {filename} -> {os.path.basename(output_path)}")

        try:
            (
                ffmpeg
                .input(file_path)
                .output(output_path, vcodec='libx264', acodec='aac', crf=23)
                .overwrite_output()
                .run()
            )
        except ffmpeg._run.Error as e:
            print(f"Error reencoding {filename}:\n{e.stderr.decode()}")
