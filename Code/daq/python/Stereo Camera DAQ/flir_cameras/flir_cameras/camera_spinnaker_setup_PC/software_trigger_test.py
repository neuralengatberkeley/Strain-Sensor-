import PySpin
import os
import time
import numpy as np
import scipy.io as sio
from datetime import datetime
import gc

# === Configuration ===
CAM_MAP = {
    '25183199': 'cam_side',
    '25185174': 'cam_top',
}
ACQUISITION_DURATION_SEC = 10

SAVE_ROOT = "C:/flir_capture"
SAVE_DIR = os.path.join(SAVE_ROOT, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Saving to {SAVE_DIR}")

# === Initialize System ===
system = PySpin.System.GetInstance()
cam_list = system.GetCameras()
cams = {}
timestamps = {}
frame_ids = {}
dimensions = {}

try:
    print(f"Number of cameras detected: {cam_list.GetSize()}")

    # === Initialize and create subfolders ===
    for serial, role in CAM_MAP.items():
        cam = cam_list.GetBySerial(serial)
        cam.Init()
        cams[serial] = cam
        timestamps[serial] = []
        frame_ids[serial] = []
        os.makedirs(os.path.join(SAVE_DIR, role), exist_ok=True)

    # === Configure trigger and acquisition ===
    for serial, cam in cams.items():
        cam.LineSelector.SetValue(PySpin.LineSelector_Line2)
        cam.LineMode.SetValue(PySpin.LineMode_Input)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
        cam.TriggerSource.SetValue(PySpin.TriggerSource_Line2)
        cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
        cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
        cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        cam.BeginAcquisition()

        # Get image size info for MATLAB
        width = cam.Width.GetValue()
        height = cam.Height.GetValue()
        dimensions[serial] = (int(height), int(width))  # rows, cols

    print("Waiting for external triggers on Line2...")

    # === Acquisition Loop ===
    start_time = time.time()
    while time.time() - start_time < ACQUISITION_DURATION_SEC:
        for serial, cam in cams.items():
            try:
                image = cam.GetNextImage(10000)
                if not image.IsIncomplete():
                    print(f"[{serial}] Pixel Format: {image.GetPixelFormatName()}")
                    print(f"{serial} resolution: {image.GetWidth()}x{image.GetHeight()}")

                    # Use human-readable timestamp format
                    timestamp_str = datetime.now().strftime("%H%M%S%f")
                    timestamps[serial].append(timestamp_str)
                    frame_ids[serial].append(image.GetFrameID())
                    print(f"[{serial}] Frame {len(frame_ids[serial])}: ID={image.GetFrameID()} Timestamp={timestamp_str}")

                    # Save image data as .raw
                    role = CAM_MAP[serial]
                    save_path = os.path.join(SAVE_DIR, role, f"{role}-{len(frame_ids[serial])-1:04d}.raw")
                    with open(save_path, 'wb') as f:
                        f.write(image.GetData())

                image.Release()
                del image
            except PySpin.SpinnakerException:
                pass

    # === Save metadata for MATLAB ===
    out = {}
    for serial in CAM_MAP:
        out[f"ts_{serial}"] = timestamps[serial]
        out[f"id_{serial}"] = frame_ids[serial]
        out[f"dims_{serial}"] = dimensions[serial]  # rows, cols
    sio.savemat(os.path.join(SAVE_DIR, "flir_data.mat"), out)
    print("Saved flir_data.mat")

finally:
    try:
        for cam in cams.values():
            if cam.IsStreaming():
                cam.EndAcquisition()
            cam.DeInit()
    except Exception as e:
        print(f"Cleanup error: {e}")

    del cam_list
    del cams
    gc.collect()
    system.ReleaseInstance()
    print("Resources released.")
