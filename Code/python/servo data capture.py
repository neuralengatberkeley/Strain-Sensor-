import os
import serial
import time
import pandas as pd
import numpy as np
import cv2
from datetime import datetime
from threading import Thread, Event
from queue import Queue

# Create a queue for storing data from the serial port
data_queue = Queue()
# Create a queue for storing image metadata
image_queue = Queue()

# Create a folder for saving images
output_folder = "dynamic_test_images"
os.makedirs(output_folder, exist_ok=True)

# Stop event to signal threads to terminate
stop_event = Event()

# Serial data reading function
def read_serial_data(ser):
    print("Serial data thread started.")
    while not stop_event.is_set():
        try:
            # Read data from serial port
            data = ser.readline().strip().decode()
            parts = data.split()
            if len(parts) == 1:  # Ensure valid data
                float_values = [float(x) for x in parts]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
                float_values.append(timestamp)  # Add timestamp
                data_queue.put(float_values)  # Add data to the queue
                print(f"Serial Data Captured: {float_values}")
                time.sleep(0.25)  # Wait 5 seconds between frames
        except Exception as e:
            print(f"Error reading serial data: {e}")
            continue

# Camera frame capturing function
def capture_images(camera):
    print("Camera thread started.")
    i = 0
    while not stop_event.is_set():
        ret, img = camera.read()
        if ret:
            # Generate timestamp and filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
            filename = os.path.join(output_folder, f"frame_{i}_{timestamp}.png")
            # Save the image
            cv2.imwrite(filename, img)
            image_queue.put((filename, timestamp))  # Add metadata to the queue
            print(f"Image Captured: {filename}")
            i += 1
        time.sleep(0.25)  # Wait 5 seconds between frames

# Main script
def main():
    # Connect to Arduino and camera
    ser = serial.Serial(port='COM4', baudrate=115200)
    camera = cv2.VideoCapture(0)
    time.sleep(2)
    ser.flush()

    # Start threads
    serial_thread = Thread(target=read_serial_data, args=(ser,))
    camera_thread = Thread(target=capture_images, args=(camera,))
    serial_thread.start()
    camera_thread.start()

    try:
        print("Press the space bar to stop.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space bar ASCII code is 32
                print("Space bar pressed. Stopping...")
                stop_event.set()  # Signal threads to stop
                break
    except KeyboardInterrupt:
        print("Keyboard interrupt. Stopping...")
        stop_event.set()

    # Wait for threads to finish
    serial_thread.join()
    camera_thread.join()

    # Release resources
    ser.close()
    camera.release()
    cv2.destroyAllWindows()

    # Process and save data
    process_and_save_data()

def process_and_save_data():
    # Collect serial data
    serial_data = []
    while not data_queue.empty():
        serial_data.append(data_queue.get())

    # Save serial data to CSV
    serial_df = pd.DataFrame(serial_data, columns=['ADC Value', 'Timestamp'])
    serial_csv = f"Bending_data_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.csv"
    serial_df.to_csv(serial_csv, index=False)
    print(f"Serial data saved to {serial_csv}")

    # Collect image metadata
    image_metadata = []
    while not image_queue.empty():
        image_metadata.append(image_queue.get())

if __name__ == "__main__":
    main()
