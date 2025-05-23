# Importing Libraries
import serial
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import cv2

# save data using pickle: https://www.geeksforgeeks.org/how-to-use-pickle-to-save-and-load-variables-in-python/

# connect to arduino and clear buffer
ser = serial.Serial(port='COM4', baudrate=115200)


# create empty array for data storage
d = []
x = str(0)  #was this causing plate to swing up initially????
ser.write(x.encode())
time.sleep(2)
ser.flush()

# vary the bend angle from 0 to 60 degrees in 10 degree increments.
# python sends a value, say 1, and arduino reads that value, multiplies the value by 10 and then subrracts 10.
# so 1 would mean 0 degrees, 2 -> 10 degrees, etc
# 7 equals to about 0 degrees pitch for IMU
#k = [7, 6, 5, 4, 3, 2, 1]
k = [0, 6]

# collect data down and up 5 times
for _ in range(101):

    for n in k:
        x = str(n)
        ser.write(x.encode())

        # This loop collects p amount of data per each angle
        p = 15
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

df.to_csv('Bending_data_abs_1p987_dyn_v2_1sec_0p08th_12_18_24.csv', index=False)
print(df)


