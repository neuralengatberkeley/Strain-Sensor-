import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Cursor
import matplotlib
import numpy as np
import re
from datetime import datetime

# Ensure interactive backend
matplotlib.use('TkAgg')

class ImagePointAnalyzer:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.columns = ['Image', 'Timestamp_image', 'Point1_x', 'Point1_y', 'Point2_x', 'Point2_y', 'Point3_x', 'Point3_y', 'Angle (degrees)']
        self.points_df = pd.DataFrame(columns=self.columns)

    def select_points(self, image_path):
        img = plt.imread(image_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.title(f"Select 3 points on: {os.path.basename(image_path)}")

        # List to store the selected points
        points = []

        # Cursor for better point selection visibility
        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)

        # Function to handle mouse clicks
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:  # Ensure valid click
                if len(points) < 3:  # Limit to 3 points
                    points.append((event.xdata, event.ydata))
                    ax.plot(event.xdata, event.ydata, 'ro')  # Mark the point
                    fig.canvas.draw()
                if len(points) == 3:  # Stop interaction after 3 points
                    plt.close()

        # Connect the click event to the figure
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()  # Blocks until the figure is closed
        return points

    def extract_timestamp(self, image_name):
        # Extract timestamp from the image name (assuming format like "image_20230101_123456_123.jpg")
        match = re.search(r'(\d{8}_\d{6}_\d{3})', image_name)
        return match.group(1) if match else None

    def collect_points(self):
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        all_points = []
        for image_file in image_files:
            image_path = os.path.join(self.image_folder, image_file)
            print(f"Processing image: {image_file}")
            selected_points = self.select_points(image_path)

            if len(selected_points) == 3:  # Ensure exactly 3 points are selected
                timestamp = self.extract_timestamp(image_file)
                points_row = {
                    'Image': image_file,
                    'Timestamp_image': timestamp,
                    'Point1_x': selected_points[0][0], 'Point1_y': selected_points[0][1],
                    'Point2_x': selected_points[1][0], 'Point2_y': selected_points[1][1],
                    'Point3_x': selected_points[2][0], 'Point3_y': selected_points[2][1],
                }
                all_points.append(points_row)
        self.points_df = pd.DataFrame(all_points)

    def calculate_angle(self, row):
        # Extract points from the row
        point1 = np.array([row['Point1_x'], row['Point1_y']])
        point2 = np.array([row['Point2_x'], row['Point2_y']])
        point3 = np.array([row['Point3_x'], row['Point3_y']])

        # Calculate vectors
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Compute dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return np.nan  # Undefined angle

        # Calculate the angle in radians
        angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

        # Convert to degrees
        angle_degrees = 180 - np.degrees(angle_radians)
        return angle_degrees

    def calculate_angles(self):
        self.points_df['Angle (degrees)'] = self.points_df.apply(self.calculate_angle, axis=1)

    def calculate_timestamp_delta(self, merged_csv_path):
        # Load the merged CSV file
        merged_df = pd.read_csv(merged_csv_path)

        # Convert timestamps to datetime objects
        merged_df['Timestamp_image'] = pd.to_datetime(merged_df['Timestamp_image'], format='%Y%m%d_%H%M%S_%f')
        merged_df['Timestamp_adc'] = pd.to_datetime(merged_df['Timestamp_adc'], format='%Y-%m-%d %H:%M:%S.%f')

        # Calculate the delta in seconds
        merged_df['Timestamp_delta'] = (merged_df['Timestamp_image'] - merged_df['Timestamp_adc']).dt.total_seconds()

        # Save the updated DataFrame back to the CSV
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Timestamp deltas calculated and saved to {merged_csv_path}")

    def merge_with_adc_data(self, adc_csv_path, output_csv_path):
        # Load ADC data
        adc_df = pd.read_csv(adc_csv_path)

        # Ensure timestamp columns are string type
        self.points_df['Timestamp_image'] = self.points_df['Timestamp_image'].astype(str)
        adc_df['Timestamp_adc'] = adc_df['Timestamp'].astype(str)

        # Concatenate dataframes without requiring a match
        merged_df = pd.concat([self.points_df.reset_index(drop=True), adc_df.reset_index(drop=True)], axis=1)

        # Save the merged dataframe to a new CSV file
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Merged data saved to {output_csv_path}")

    def save_to_csv(self, file_path):
        self.points_df.to_csv(file_path, index=False)

    def analyze(self):
        self.collect_points()
        self.calculate_angles()
        print(self.points_df)


# Example usage:
image_folder = 'C:/Users/toppe/OneDrive - CSU Maritime Academy/Documents/GitHub/Strain-Sensor-/Code/python/dynamic_test_images'
adc_csv_path = 'C:/Users/toppe/OneDrive - CSU Maritime Academy/Documents/GitHub/Strain-Sensor-/Code/python/Bending_data_20250313_132957_604.csv'  # Replace with the actual path
output_csv_path = 'merged_data_with_angles_and_adc.csv'

analyzer = ImagePointAnalyzer(image_folder)
analyzer.analyze()
analyzer.save_to_csv('points_data_with_timestamps.csv')
analyzer.merge_with_adc_data(adc_csv_path, output_csv_path)
analyzer.calculate_timestamp_delta(output_csv_path)
