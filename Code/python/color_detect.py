import cv2
import numpy as np
import pandas as pd

# Initialize webcam input
vid = cv2.VideoCapture(0)

# Initialize a dataframe to store centroids
centroid_data = pd.DataFrame()
frame_count = 0

while True:
    # Capture the current frame
    ret, frame = vid.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold to detect dark regions
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (descending) and select the top three
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    # Initialize a dictionary for the current frame's data
    frame_data = {"frame": frame_count}

    # Loop through the top three contours and save centroid data
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the centroid of the rectangle
        centroid_x = x + w // 2
        centroid_y = y + h // 2

        # Store centroid data in the dictionary
        frame_data[f"xpos{i+1}"] = centroid_x
        frame_data[f"ypos{i+1}"] = centroid_y

    # Append the frame data to the dataframe
    centroid_data = pd.concat([centroid_data, pd.DataFrame([frame_data])], ignore_index=True)

    # Display the current frame
    cv2.imshow("frame", frame)

    # Increment the frame count
    frame_count += 1

    # Break the loop if 'q' is pressedq
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the centroid data to a CSV file
centroid_data.to_csv("centroid_data.csv", index=False)

# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
