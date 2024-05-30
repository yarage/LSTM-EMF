import os
import pandas as pd
from shutil import move

# Function to check if the data is empty or incomplete
def check_data(file_path):
    try:
        column_names = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
        data = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, names=column_names)
        
        # Check for empty data
        if data.empty:
            return 'empty'
        
        # Check for incomplete data
        if len(data) < 1440:  # Assuming 1440 rows for a complete day
            return 'incomplete'
        if len(data) == 1:
            return 'empty'
        
        return 'correct'
    
    except pd.errors.EmptyDataError:
        return 'empty'
    except Exception as e:
        return 'incomplete'

# Folder containing .min files
folder_path = r'C:\\Users\\PRACTICAS - IGP\\Documents\\LSTM2-INITIAL\\RAW_2010-2020(decompressed)'

# Create folders for each category
empty_folder = os.path.join(folder_path, 'empty_data')
incomplete_folder = os.path.join(folder_path, 'incomplete_data')
correct_folder = os.path.join(folder_path, 'complete_data')

# Ensure the folders exist
for folder in [empty_folder, incomplete_folder, correct_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Get a list of all .min files in the folder
min_files = [f for f in os.listdir(folder_path) if f.endswith('.min')]

# Iterate through each .min file
for min_file in min_files:
    file_path = os.path.join(folder_path, min_file)
    
    # Check the status of the data
    data_status = check_data(file_path)
    
    # Move the file to the corresponding folder
    if data_status == 'empty':
        move(file_path, os.path.join(empty_folder, min_file))
    elif data_status == 'incomplete':
        move(file_path, os.path.join(incomplete_folder, min_file))
    elif data_status == 'complete':
        move(file_path, os.path.join(correct_folder, min_file))
