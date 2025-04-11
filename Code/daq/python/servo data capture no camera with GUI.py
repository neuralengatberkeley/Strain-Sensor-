import serial
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from datetime import datetime
import tkinter as tk
from tkinter import ttk

# Add path to Code/analysis/
analysis_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "analysis"))
if analysis_path not in sys.path:
    sys.path.insert(0, analysis_path)

from analysis import bender_class
from config import path_to_repository
from sklearn.model_selection import train_test_split

# --- GUI Input ---
def get_user_input():
    root = tk.Tk()
    root.title("Test Metadata")
    root.geometry("600x400")  # Doubled size
    root.resizable(False, False)

    values = {}

    def submit():
        values['sample_length'] = entry_length.get()
        values['testing_type'] = combo_type.get()
        values['version'] = "v" + entry_version.get()
        root.destroy()

    tk.Label(root, text="Sample Length (e.g. 1.86)").pack(pady=10)
    entry_length = tk.Entry(root, font=("Arial", 14))
    entry_length.pack(pady=10)

    tk.Label(root, text="Testing Type").pack(pady=10)
    combo_type = ttk.Combobox(root, values=["static", "misalignment", "reapplication"], state="readonly", font=("Arial", 14))
    combo_type.current(0)
    combo_type.pack(pady=10)

    tk.Label(root, text="Version Number (e.g. 2)").pack(pady=10)
    entry_version = tk.Entry(root, font=("Arial", 14))
    entry_version.pack(pady=10)

    tk.Button(root, text="Submit", command=submit, font=("Arial", 14)).pack(pady=20)
    root.mainloop()

    return values['sample_length'], values['testing_type'], values['version']

# Get user input
sample_length, testing_type, version = get_user_input()

# Connect to Arduino
ser = serial.Serial(port='COM4', baudrate=115200)
d = []
x = str(0)
ser.write(x.encode())
time.sleep(2)
ser.flush()

k = [0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]
for _ in range(5):
    for n in k:
        x = str(n)
        ser.write(x.encode())
        p = 50
        for _ in range(p):
            data = ser.readline().strip().decode()
            parts = data.split()
            if len(parts) != 4 or (len(parts) > 0 and parts[0] == '0'):
                continue
            float_values = []
            for m in parts:
                try:
                    float_values.append(float(m))
                except ValueError:
                    float_values.append(100)
            print(float_values)
            d.append(float_values)

ser.close()

# Create DataFrame
df = pd.DataFrame(d)
df.columns = ['Theoretical Angle (deg)', 'IMU Angle (deg)', 'ADC Value', 'Rotary Encoder']

# Save CSV
today = datetime.today()
today_str = f"{today.month}_{today.day}_{today.strftime('%y')}"
# Absolute path to your CSV Data folder
# Hardcode your target path
csv_data_dir = r"C:/Users/toppe/OneDrive - CSU Maritime Academy/Documents/GitHub/Strain-Sensor-/CSV Data"

#script_dir = os.path.dirname(os.path.abspath(__file__))
#csv_data_dir = os.path.abspath(os.path.join(script_dir, "..", "csv_data"))
folder_path = os.path.join(csv_data_dir, today_str)
os.makedirs(folder_path, exist_ok=True)

filename = f"{sample_length}_{testing_type}_{version}_{today_str}.csv"
full_path = os.path.join(folder_path, filename)
df.to_csv(full_path, index=False)
print(f"Data saved to: {full_path}")

# Plot
g = bender_class()
g.load_data(full_path)
g.normalize_adc_over_R0()
g.plot_data(scatter=True)
plt.show()
