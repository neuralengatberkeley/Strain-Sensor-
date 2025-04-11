# Importing Libraries
import serial
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import cv2
import os
from datetime import datetime

import sys

# Get absolute path to Code/analysis/
analysis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "analysis"))

# Add analysis folder to sys.path
if analysis_path not in sys.path:
    sys.path.insert(0, analysis_path)

# Now you can import both modules directly from that folder
from analysis import bender_class
from config import path_to_repository

from sklearn.model_selection import train_test_split

# save data using pickle: https://www.geeksforgeeks.org/how-to-use-pickle-to-save-and-load-variables-in-python/

# connect to arduino and clear buffer
ser = serial.Serial(port='COM4', baudrate=115200)


# create empty array for data storage
d = []
x = str(0)  #was this causing plate to swing up initially????
ser.write(x.encode())
time.sleep(2)
ser.flush()

# vary the bend angle from 0 to 60 degrees in 15 degree increments.
# python sends a value, say 1, and arduino reads that value, multiplies the value by 10 and then subrracts 10.
# so 1 would mean 0 degrees, 2 -> 10 degrees, etc
# 7 equals to about 0 degrees pitch for IMU
#k = [7, 6, 5, 4, 3, 2, 1]
k = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]

# collect data down and up 5 times
for _ in range(5):

    for n in k:
        x = str(n)
        ser.write(x.encode())

        # This loop collects p amount of data per each angle
        p = 50
        for _ in range(p):
            data = ser.readline().strip().decode()
            # Attempt to split data by tabs and spaces, then take only the relevant parts
            parts = data.split()
            # Ensure exactly 3 parts for each row
            if len(parts) != 4 or (len(parts) > 0 and parts[0] == '0'):
                continue  # Skip this iteration if data is not as expected
            # Convert parts to floats, replace with 100 if conversion fails
            float_values = []
            for m in parts:
                try:
                    float_values.append(float(m))
                except ValueError:
                    float_values.append(100)  # Default value if conversion fails
            print(float_values)
            d.append(float_values)  # Append the row of floats to list d

ser.close()

# convert list d to pandas data frame and save as csv
df = pd.DataFrame(d)
df.columns = ['Theoretical Angle (deg)', 'IMU Angle (deg)', 'ADC Value', 'Rotary Encoder']
#df.replace([np.inf, -np.inf], np.nan, inplace=True)
#df.dropna(how="all", inplace=True)

######################################################################


sample_length = "1.86"         # or similar
testing_type = "static"         # or similar
version = "v1"

# --- AUTOMATED SAVE SECTION ---

# Format today's date as 'month_day_year'
today = datetime.today()
today_str = f"{today.month}_{today.day}_{today.strftime('%y')}"

# Define base directory and folder path
# Absolute path to this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to csv_data folder (assumed to be alongside the Code/ folder)
csv_data_dir = os.path.abspath(os.path.join(script_dir, "..", "csv_data"))

# Full folder path for today’s date
folder_path = os.path.join(csv_data_dir, today_str)

# Create folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

# Create filename (no timestamp)
filename = f"{sample_length}_{testing_type}_{version}_{today_str}.csv"
full_path = os.path.join(folder_path, filename)

# Save the DataFrame
df.to_csv(full_path, index=False)
print(f"Data saved to: {full_path}")


#  Plotting normalized data delta R over Ro vs bend angle

g = bender_class()
g.load_data(full_path)
g.normalize_adc_over_R0()
g.plot_data(scatter=True)

plt.show()  # ← This forces the plot to appear
