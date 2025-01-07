import cv2
import os
import numpy as np

# Initialize variables
image_folder = "C:/Users/toppenheim/Desktop/UCSF/Preeya UCB/Strain-Sensor-/Code/snapshots/"  # Replace with your image folder path
output_folder = "segmented_objects"  # Folder to save segmented masks

# Check if the output folder exists, create if not
os.makedirs(output_folder, exist_ok=True)

# Colors for circles (in BGR format)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Green, Red, Yellow
classes = [1, 2, 3, 4]  # Corresponding class labels for the circles


def draw_circle(event, x, y, flags, param):
    """Mouse callback function to draw circles on the image and update the mask."""
    global image, mask, circle_count

    if event == cv2.EVENT_LBUTTONDOWN and circle_count < 4:  # Left mouse click
        color = colors[circle_count]
        class_label = classes[circle_count]

        # Draw the circle on the image (visualization)
        cv2.circle(image, (x, y), 20, color, -1)

        # Draw the circle on the mask (using class label)
        cv2.circle(mask, (x, y), 20, class_label, -1)

        circle_count += 1
        print(f"Circle {circle_count} added at ({x}, {y}) with class {class_label}")


# Iterate over files in the directory
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)

    # Read the image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Warning: Unable to read {image_path}. Skipping.")
        continue  # Skip files that cannot be read

    # Clone the image and create an empty mask
    image = original_image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Single-channel mask
    circle_count = 0  # Reset circle counter for each image

    # Set up the window and mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_circle)

    print(f"Processing {filename}. Click to add 4 circles.")
    while True:
        # Display the image
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        # Press 's' to save and proceed to the next image
        if key == ord('s') and circle_count == 4:
            print(f"Saving segmented mask for {filename}.")

            # Save the mask (used for segmentation)
            mask_path = os.path.join(output_folder, f"mask_{filename}")
            cv2.imwrite(mask_path, mask)

            # Optionally save the annotated image (for reference)
            annotated_image_path = os.path.join(output_folder, f"annotated_{filename}")
            cv2.imwrite(annotated_image_path, image)
            break

        # Press 'q' to quit early
        if key == ord('q'):
            print("Quitting...")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()  # Close the window after processing each image
