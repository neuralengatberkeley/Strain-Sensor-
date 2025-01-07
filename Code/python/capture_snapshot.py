import cv2
import os
from datetime import datetime

# Specify the folder to save snapshots
SAVE_FOLDER = "snapshots"

# Create the folder if it doesn't exist
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press the space bar to take a snapshot. Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the video feed
    cv2.imshow("Webcam", frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Space bar key
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_FOLDER, f"orig_finger_{timestamp}.png")

        # Save the snapshot
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")

    elif key == ord('q'):  # Quit key
        print("Exiting...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()