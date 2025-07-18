from scipy.io import loadmat
from tkinter import Tk, filedialog

# === Hide the main tkinter window ===
Tk().withdraw()

# === Ask user to select a .mat file ===
file_path = filedialog.askopenfilename(
    title="Select a .mat file",
    filetypes=[("MATLAB files", "*.mat")]
)


data = loadmat(file_path)
print(data.keys())  # View variable names

# Access a variable
#my_var = data['variable_name']  # Replace with actual variable name
