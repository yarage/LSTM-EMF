import os

# Set environment variable to avoid OpenMP runtime error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import random
#from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt





def extract_info_from_filename(file_name):
    match = re.match(r'([a-zA-Z]+)_(\d{6})\.min', file_name)
    if match:
        station_code, date_str = match.groups()
        year = int(date_str[:2])
        month = int(date_str[2:4])
        return station_code, year, month
    else:
        raise ValueError(f"Invalid file name format: {file_name}")

def load_data(folder_path):
    column_mapping = {
        'D(Deg)': 'D(deg)',
        'D'     : 'D(deg)',
        'D(deg)': 'D(deg)',
        'H(nT)' : 'H(nT)' ,
        'H'     : 'H(nT)' ,
        'Z(nT)' : 'Z(nT)' ,
        'Z'     : 'Z(nT)' ,
        'I(Deg)': 'I(deg)',
        'I'     : 'I(deg)',
        'I(deg)': 'I(deg)',
        'F(nT)' : 'F(nT)' ,
        'F'     : 'F(nT)'
    }
    
    read_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.min')]

    data_frames = []
    for file_path in read_files:
        station_code, year, month = extract_info_from_filename(os.path.basename(file_path))
        for separator in ['\s+', '\s{2,}']:  # Try different spacing formats
            try:
                df = pd.read_table(file_path, skiprows=2, sep=separator)
                break  # Stop trying other separators if successful
            except Exception as e:
                print(f"Error reading file {file_path} with separator '{separator}': {e}")
                continue
        
        if not df.empty:
            # Check if the columns exist before dropping
            columns_to_drop = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'HH', 'MM.1']  # Remove both 'hh mm' and 'HH MM' columns
            existing_columns = set(df.columns)
            columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

            # Drop the unwanted columns
            df = df.drop(columns=columns_to_drop, errors='ignore')

            # Rename columns based on the mapping dictionary
            df.rename(columns=column_mapping, inplace=True)

            # Remove any additional columns that are not part of the standard magnetic data columns
            standard_columns = ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
            non_standard_columns = [col for col in df.columns if col not in standard_columns]
            if non_standard_columns:
                #print(f"Ignoring non-standard columns: {non_standard_columns}")
                df = df.drop(columns=non_standard_columns, errors='ignore')
                # Rename columns based on the mapping dictionary
                df.rename(columns=column_mapping, inplace=True)

            df['Station'] = station_code
            df['Year'] = year
            df['Month'] = month
            
            data_frames.append(df)

    return data_frames


def print_data_frames(data_frames):
    for i, df in enumerate(data_frames):
        print(f"DataFrame {i + 1} - Shape: {df.shape}")
        print(df.head())
        print("----" * 20)






def create_sequences(df, columns, sequence_length, scaler):
    #output_seq_length = 1  # Predict the next single value for each column

    # Extract columns D, H, and Z from the DataFrame
    raw_data = df[columns].values

    # Apply MinMaxScaler
    data = scaler.fit_transform(raw_data)

    # Initialize empty lists to store input sequences (X) and target values (y)
    X_sequences, y_targets, x_orig = [], [], []

    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length + 1):
        # Extract sequences of length sequence_length as input (X)
        X_seq = data[i:i + sequence_length]
        # Extract the next value as the target (y)
        #if i == 0 :
            #original_input = scaler.inverse_transform(X_seq)
            #print(X_seq)
            #print(original_input)
        x_orig.append(X_seq)

    
        # Define the range for selecting the central point
        central_point_range = (len(X_seq) // 2) - 2, (len(X_seq) // 2) + 2
        
        # Select a random central point
        central_point = random.randint(central_point_range[0], central_point_range[1])
        
        # Get the first value of the curve
        first_value = X_seq[0]


        min_change = []
        max_change = []

        # Calculate the range for the new value (1% to 25% of the first value)
        for i in range(len(columns)):
            min_change.append(0.01 * first_value[i])
            max_change.append(0.25 * first_value[i])

        change_value = []
        
        # Generate a random value within the range
        for i in range(len(columns)):
            sign = 1
            if random.random() < 0.5:
                sign *= -1
            
            change_value.append(sign * random.uniform(min_change[i], max_change[i]))
        
        # Modify the curve from the central point to the end
        X_seq = np.array(X_seq)
        for i in range(len(columns)):
            for j in range(central_point, sequence_length):
                X_seq[j][i] =  X_seq[j][i] + change_value[i]
        
        #for i in range(central_point, len(X_seq)):
            

            #curve_values[i] += change_value
        
        #return curve_values, change_value, central_point

        y_tar = []
        y_tar.append(change_value)
        y_tar[0].append(central_point)

        #print(X_seq)
        #print(y_tar)

        #y_target = y_tar[0] + [y_tar[1]]
        y_target = y_tar
    



        #y_target = data[i + sequence_length:i + sequence_length + output_seq_length]

        # Append the sequences to the lists
        X_sequences.append(X_seq)
        y_targets.append(y_target)

    # Convert lists to NumPy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    return X_sequences, y_targets, x_orig




    #### to TEST

def prepare_evaluation_data(folder_path, columns, sequence_length, scaler):
    # Load the evaluation data
    eval_data_frames = load_data(folder_path)

    # Initialize empty lists to store sequences for all DataFrames
    all_X_sequences, all_y_targets, all_x_orig = [], []

    # Iterate through each DataFrame to create sequences
    for df in eval_data_frames:
        #create_sequences(df, columns, sequence_length, scaler)
        X_sequences, y_targets, x_orig = create_sequences(df, columns, sequence_length, scaler)
        all_X_sequences.append(X_sequences)
        all_y_targets.append(y_targets)
        all_x_orig.append(x_orig)

    # Concatenate sequences from all DataFrames
    concatenated_X_sequences = np.concatenate(all_X_sequences)
    concatenated_y_targets = np.concatenate(all_y_targets)
    concatenated_x_orig = np.concatenated(all_x_orig)

    # Print the shapes of the concatenated sequences for verification
    #print("Concatenated X Sequences Shape:", concatenated_X_sequences.shape)
    #print("Concatenated y Targets Shape:", concatenated_y_targets.shape)

    return concatenated_X_sequences, concatenated_y_targets, concatenated_x_orig



def load_data_single_file(file_path):

    column_mapping = {
        'D(Deg)': 'D(deg)',
        'D'     : 'D(deg)',
        'D(deg)': 'D(deg)',
        'H(nT)' : 'H(nT)' ,
        'H'     : 'H(nT)' ,
        'Z(nT)' : 'Z(nT)' ,
        'Z'     : 'Z(nT)' ,
        'I(Deg)': 'I(deg)',
        'I'     : 'I(deg)',
        'I(deg)': 'I(deg)',
        'F(nT)' : 'F(nT)' ,
        'F'     : 'F(nT)'
    }
    

    
    station_code, year, month = extract_info_from_filename(os.path.basename(file_path))
    df = pd.read_table(file_path, skiprows=2, sep = '\s+')
    if not df.empty:
        # Check if the columns exist before dropping
        columns_to_drop = ['DD', 'MM', 'YYYY', 'hh', 'mm']
        existing_columns = set(df.columns)
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

        # Drop the unwanted columns
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Rename columns based on the mapping dictionary
        df.rename(columns=column_mapping, inplace=True)

        df['Station'] = station_code
        df['Year'] = year
        df['Month'] = month

    return df

def evaluate_single_file(model, file_path, columns, sequence_length, scaler):
    # Load the single file data
    single_file_df = load_data_single_file(file_path)

    filename = single_file_df['Station'].iloc[0]  # Assuming 'Station' contains the station code

    # Extract a single magnitude from the DataFrame
    original_data = single_file_df[columns].values    

    # Create sequences for the single file
    X_sequences, y_targets, x_orig = create_sequences(single_file_df, columns, sequence_length, scaler)
    #create_sequences(df, columns, sequence_length, scaler)
    
   
    input_sequences = X_sequences
    output_sequences = y_targets


    input_seq_length = input_sequences.shape[1]
    output_seq_length = output_sequences.shape[1]
    

    # Define input and output dimensions
    input_dim = input_sequences.shape[2]
    output_dim = output_sequences.shape[2]

    latent_dim = 64


    predictions_scaled = model.predict(input_sequences)
    print("Predictions_scaled:")
    print(predictions_scaled.shape)
    print(predictions_scaled)
    
    #print("Predicted output shape: ", predictions_scaled.shape)
    
    predictions_scaled = predictions_scaled.squeeze().tolist()

    #print(predictions_scaled[0])
    
    y_targets = y_targets.squeeze().tolist()

    # Assuming the example shape is (num_sequences, sequence_length, num_features)
    example_shape = (len(input_sequences), 1, 3)  # Example shape from your data

    # Create an array filled with zeros with the same shape as the example
    base = np.zeros_like(np.zeros(example_shape))


    for sublist in predictions_scaled:
        sublist[:3] = scaler.inverse_transform([sublist[:3]])[0]

    for sublist in y_targets:
        sublist[:3] = scaler.inverse_transform([sublist[:3]])[0]

    for k in range(2):
        x_input = scaler.inverse_transform(input_sequences[k])

        base_inversed = scaler.inverse_transform(base[k])

        original_input = scaler.inverse_transform(x_orig[k])
    
        print("Original_input:")
        print(original_input)
        print("x_input:")
        print(x_input)
          
        print("FINAL")
        A = []
        B = []
        C = []
        A = predictions_scaled[k][0:3]
        B = y_targets[k][0:3]
        C = base_inversed[0]

        print("A: (prediction_Scaled)")
        print(A)
        print("B: (y_Targets)")        
        print(B)
        print("C: (base)")
        print(C)
        
        offset_pred = [x - y for x, y in zip(A, C)]
        #offset_pred = offset_pred[0]
        print("offset_pred")
        print(offset_pred)

        offset_actual = [x - y for x, y in zip(B, C)]
        #offset_actual = offset_actual[0]
        print("offset_actual")
        print(offset_actual)

        print("Change position:")
        position = y_targets[k][3] 
        print(position)

        #plot_subplots_multiple(predictions, y_targets, sequence_length, filename)
        plot_subplots(original_input, x_input, sequence_length, 'Data sin salto', 'Data con salto, usada como input')

        corrected = add_values_from_array(x_input, offset_pred, position)
        corr_act = add_values_from_array(x_input, offset_actual, position)

        print("Corrected")
        print(corrected)

        diff_pred = original_input - corrected
        diff_act = original_input - corr_act

        
        '''
        plot_subplots(x_input, corrected, sequence_length, 'data con salto', 'data corregida')
        plot_subplots(x_input, corr_act, sequence_length, 'data con salto', 'data ideal')

        plot_subplots(diff_pred, diff_act, sequence_length, 'Dif input-corr', 'Dif input-corr ideal')
        plot_subplots(original_input, diff_act, sequence_length, 'data con salto', 'Dif input-corr ideal')
        '''

        #y_ranges = [(-4.60,-4.40),(22850,22940),(-4070,4030)]
        
        #plot_subplots(original_input, corrected, sequence_length, 'data con salto', 'data corregida')
        #plot_subplots(original_input, corr_act, sequence_length, 'data con salto', 'data ideal')
        plot_subplots(original_input, corrected, sequence_length, 'Data Objetivo', 'data modelo')
        plot_subplots3(original_input, corrected, x_input, sequence_length, 'Data Objetivo', 'data modelo', 'input')

                
    return


def add_values_from_array(data, array, position):
    data1 = np.copy(data)
    # Starting from the position provided, add the values from the array to the data
    for i in range(int(position), len(data1)):
        data1[i] -= array
    return data1


def plot_subplots(data1, data2, sequence_length, data1label, data2label, y_ranges=None):
    num_cols = data1.shape[1]  # Get the number of columns in the data

    # Create subplots with a layout determined by the number of columns
    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 8), sharex=True)

    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)']
    
    # Plot each column in a separate subplot
    for i, magnitude in enumerate(magnitudes):
        x_values = np.arange(sequence_length)  # Adjust x-axis values
        axes[i].plot(x_values, data1[:, i], label=data1label, color='blue')
        axes[i].plot(x_values, data2[:, i], label=data2label, color='red', linestyle='--')
        axes[i].set_ylabel(magnitude)
        axes[i].legend()
        
        # Set y-axis range if provided
        if y_ranges is not None and y_ranges[i] is not None:
            axes[i].set_ylim(y_ranges[i])

    plt.tight_layout()
    plt.show()

def plot_subplots3(data1, data2, data3, sequence_length, data1label, data2label, data3label, y_ranges=None):
    num_cols = data1.shape[1]  # Get the number of columns in the data

    # Create subplots with a layout determined by the number of columns
    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 8), sharex=True)

    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)']
    
    # Plot each column in a separate subplot
    for i, magnitude in enumerate(magnitudes):
        x_values = np.arange(sequence_length)  # Adjust x-axis values
        axes[i].plot(x_values, data1[:, i], label=data1label, color='blue')
        axes[i].plot(x_values, data2[:, i], label=data2label, color='red', linestyle='--')
        axes[i].plot(x_values, data3[:, i], label=data3label, color='orange', linestyle='-.')
        axes[i].set_ylabel(magnitude)
        axes[i].legend()
        
        # Set y-axis range if provided
        if y_ranges is not None and y_ranges[i] is not None:
            axes[i].set_ylim(y_ranges[i])

    plt.tight_layout()
    plt.show()


def plot_subplots_multiple(data1, data2, sequence_length, filename):
    num_cols = data1.shape[1]  # Get the number of columns in the data

    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 8), sharex=True)

    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)']

    # Plot each column of data1 and data2 in a separate subplot
    for i, magnitude in enumerate(magnitudes):
        x_values = np.arange(sequence_length, len(data1) + sequence_length)  # Adjust x-axis values
        axes[i].plot(x_values, data1[:, i], label='Prediction', color='blue')
        axes[i].plot(x_values, data2[:, i], label='Actual', color='red')
        #axes[i].set_title(f'Column {i+1}')
        #axes[i].set_xlabel('Index')
        axes[i].set_ylabel(magnitude)
        axes[i].legend()

    # Set common labels and title
    axes[-1].set_xlabel('Time')
    plt.suptitle(f'Evolution of Magnitudes Over Time - {filename}')


    plt.tight_layout()
    plt.show()




def evaluate_folder(folder_path, sequence_length, columns, model, scaler):
    # Get a list of all .min files in the folder
    min_files = [f for f in os.listdir(folder_path) if f.endswith('.min')]
    # Loop through each .min file
    for min_file in min_files:
        file_path = os.path.join(folder_path, min_file)
        evaluate_single_file(model, file_path, columns, sequence_length, scaler)


    return

    
def main():

    # List of magnitude column names
    columns = ['D(deg)', 'H(nT)', 'Z(nT)']
        
    # Construct model filename based on user input
    model_filename = f'LSTM3_offset_model.h5'

    # Create MinMaxScaler and load model
    scaler = MinMaxScaler()
    #column_name = f'{get_column}'
    model = load_model(model_filename)

    #print("Input shape: ", model.input_shape)
    #print("Output shape: ", model.output_shape)

    # Define a dictionary to map each magnitude to its threshold
    #threshold_dict = {'D': 0.25, 'H': 3.5, 'Z': 1, 'I': 0.0025, 'F': 4.5}
    #anomaly_threshold = [0.25, 3.5, 1]

    # Get the threshold for the current magnitude
    #anomaly_threshold = threshold_dict[magnitude_input]

    # Set other parameters
    sequence_length = 10
    folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM2-Train\\TEST_SET'

    # Call evaluate_folder with user-defined parameters
    evaluate_folder(folder_path, sequence_length, columns, model, scaler)

    
    
if __name__ == "__main__":
    main()

