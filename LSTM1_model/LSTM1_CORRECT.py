import os
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
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
    read_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.min')]

    data_frames = []
    for file_path in read_files:
        station_code, year, month = extract_info_from_filename(os.path.basename(file_path))
        df = pd.read_table(file_path, skiprows=2, sep = '\s+')
        if not df.empty:
            # Check if the columns exist before dropping
            columns_to_drop = ['DD', 'MM', 'YYYY', 'hh', 'mm']
            existing_columns = set(df.columns)
            columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

            # Drop the unwanted columns
            df = df.drop(columns=columns_to_drop, errors='ignore')

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


def create_sequences(df, column_name, sequence_length, scaler):
    # Extract 'D' (declination) column from the DataFrame
    raw_data = df[column_name].values

    # Reshape the 1D array to a 2D array
    raw_data_reshaped = raw_data.reshape(-1, 1)

    # Apply MinMaxScaler
    data = scaler.fit_transform(raw_data_reshaped)

    # Initialize empty lists to store input sequences (X) and target values (y)
    X_sequences, y_targets = [], []

    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length):
        # Extract a sequence of length sequence_length as input (X)
        X_seq = data[i:i + sequence_length]
        # Extract the next value as the target (y)
        y_target = data[i + sequence_length]
        
        # Append the sequences to the lists
        X_sequences.append(X_seq)
        y_targets.append(y_target)

    # Convert lists to NumPy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    return X_sequences, y_targets


    #### to TEST

def prepare_evaluation_data(folder_path, column_name, sequence_length):
    # Load the evaluation data
    eval_data_frames = load_data(folder_path)

    # Initialize empty lists to store sequences for all DataFrames
    all_X_sequences, all_y_targets = [], []

    # Iterate through each DataFrame to create sequences
    for df in eval_data_frames:
        X_sequences, y_targets = create_sequences(df, column_name, sequence_length, scaler)
        all_X_sequences.append(X_sequences)
        all_y_targets.append(y_targets)

    # Concatenate sequences from all DataFrames
    concatenated_X_sequences = np.concatenate(all_X_sequences)
    concatenated_y_targets = np.concatenate(all_y_targets)

    return concatenated_X_sequences, concatenated_y_targets



def load_data_single_file(file_path):
    station_code, year, month = extract_info_from_filename(os.path.basename(file_path))
    df = pd.read_table(file_path, skiprows=2, sep = '\s+')
    if not df.empty:
        # Check if the columns exist before dropping
        columns_to_drop = ['DD', 'MM', 'YYYY', 'hh', 'mm']
        existing_columns = set(df.columns)
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

        # Drop the unwanted columns
        df = df.drop(columns=columns_to_drop, errors='ignore')

        df['Station'] = station_code
        df['Year'] = year
        df['Month'] = month

    return df

def evaluate_single_file(model, file_path, column_name, sequence_length, anomaly_threshold, scaler):

    # Load the single file data
    single_file_df = load_data_single_file(file_path)

    # Extract a single magnitude from the DataFrame
    original_data = single_file_df[column_name].values    

    # Create sequences for the single file
    X_sequences, y_targets = create_sequences(single_file_df, column_name, sequence_length, scaler)

    # Reshape X_sequences to match the input shape expected by LSTM
    X_sequences_reshaped = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))

    # Predict using the trained model
    predictions_scaled = model.predict(X_sequences_reshaped)

    # Inverse transform the scaled predictions
    predictions = scaler.inverse_transform(predictions_scaled)

    # Initialize a list to store corrected values
    corrected_values = []

    # Initialize a flag to indicate if the previous value was replaced
    prev_replaced = False

    # Calculate the absolute differences between actual and predicted values
    differences = np.abs(y_targets.flatten() - predictions.flatten())

    # Identify anomalies based on the threshold
    anomalies = differences > anomaly_threshold

    # Iterate through the sequences
    for i in range(len(y_targets)):
        # Check if the previous value was replaced
        if prev_replaced:
            # Append the predicted value to the corrected list
            corrected_values.append(predictions[i][0])
            # Reset the flag
            prev_replaced = False
        else:
            # Append the original value if no anomaly
            corrected_values.append(y_targets[i][0])

        # Calculate the absolute difference between actual and predicted values
        diff = np.abs(corrected_values[-1] - predictions[i][0])

        # Check if the difference exceeds the threshold
        if diff > anomaly_threshold:
            # Replace the last corrected value with the predicted value
            corrected_values[-1] = predictions[i][0]
            # Set the flag to indicate replacement
            prev_replaced = True

    # Share x-axis across subplots
    plt.subplots(4, 1, sharex=True)

    # Plot actual vs predicted values
    plt.subplot(4, 1, 1)
    plt.plot(original_data, label='Actual')
    plt.plot(np.arange(sequence_length, len(original_data)), predictions.flatten(), label='Predicted')
    plt.legend()
    plt.title(f'Actual vs Predicted - {os.path.basename(file_path)}')

    # Plot the differences
    plt.subplot(4, 1, 2)
    plt.plot(original_data, label='Actual')
    plt.legend()
    plt.title(f'Actual Data - {os.path.basename(file_path)}')

    # Plot anomalies
    plt.subplot(4, 1, 3)
    # Plot the differences
    x_values = np.arange(sequence_length, sequence_length + len(differences))
    plt.plot(x_values, differences, label='Differences')
    plt.axhline(anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')

    # Highlight differences exceeding the threshold
    anomalies_indices = np.where(anomalies)[0]
    plt.scatter(anomalies_indices + sequence_length, differences[anomalies_indices], c='red', label='Anomalies')
    plt.legend()
    plt.title(f'Anomalies Detected - {os.path.basename(file_path)}')

    # Plot corrected values
    plt.subplot(4, 1, 4)
    plt.plot(original_data[:sequence_length], label='Original Data')
    plt.plot(np.arange(sequence_length, len(corrected_values) + sequence_length), corrected_values, label='Corrected Data')
    plt.legend()
    plt.title(f'Corrected Data - {os.path.basename(file_path)}')

    plt.tight_layout()
    plt.show()

    return

def evaluate_folder(folder_path, sequence_length, anomaly_threshold, column_name, model, scaler):
    # Get a list of all .min files in the folder
    min_files = [f for f in os.listdir(folder_path) if f.endswith('.min')]

    # Loop through each .min file
    for min_file in min_files:
        file_path = os.path.join(folder_path, min_file)
        evaluate_single_file(model, file_path, column_name, sequence_length, anomaly_threshold, scaler)


    return


def correct_data_with_lstm(model, file_path, column_name, sequence_length, anomaly_threshold, scaler):
    #(model, file_path, column_name, sequence_length, anomaly_threshold, scaler):

    # Load the single file data
    single_file_df = load_data_single_file(file_path)

    # Extract a single magnitude from the DataFrame
    original_data = single_file_df[column_name].values
 
    # Initialize a list to store corrected values
    corrected_values = []

    # Copy the initial sequence_length values to the corrected list
    #corrected_values.extend(data[:sequence_length])
    corrected_values.extend(original_data[:sequence_length])

    # Initialize a flag to indicate if the previous value was replaced
    prev_replaced = False

    anomalies_indices = []

    # Iterate through the data starting from sequence_length
    for i in range(sequence_length, len(original_data)):
        # Extract the last sequence_length values from the corrected list
        input_sequence = corrected_values[-sequence_length:]

        # Convert input_sequence to a NumPy array
        input_sequence = np.array(input_sequence)
        input_sequence_shaped = input_sequence.reshape(-1,1)
        
        

        # To scale
        input_sequence_scaled = scaler.fit_transform(input_sequence_shaped)

        # Reshape the input sequence to match the input shape expected by LSTM
        input_sequence_reshaped = np.array(input_sequence_scaled).reshape((1, sequence_length, 1))

        # Predict using the trained model
        prediction_scaled = model.predict(input_sequence_reshaped)

        # Inverse transform the scaled prediction
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        # Calculate the absolute difference between actual and predicted values
        diff = np.abs(original_data[i] - prediction)

        # Check if the difference exceeds the threshold
        if diff > anomaly_threshold:
            # Replace the actual value with the predicted value
            corrected_values.append(prediction)
            # Set the flag to indicate replacement
            prev_replaced = True
            # Append the index to the anomalies_indices list
            anomalies_indices.append(i)
        else:
            # Maintain the actual value if no anomaly
            corrected_values.append(original_data[i])
            # Reset the flag
            prev_replaced = False

    return corrected_values, anomalies_indices




def plot_results(original_data, corrected_values, differences, anomalies_indices, anomaly_threshold, sequence_length, file_path, magnitude_input):
    #file_path = os.path.join(file_path)

    """
    Plot original data, corrected data, differences, and anomalies.
    """
    plt.subplots(4, 1, sharex=True, figsize=(12, 8))

    # Plot actual vs predicted values
    plt.subplot(4, 1, 1)
    plt.plot(original_data, label='Actual')
    plt.legend()
    plt.title(f'Actual Data - Magnitude: {magnitude_input}, File: {os.path.basename(file_path)}')

    # Plot corrected values
    plt.subplot(4, 1, 2)
    #plt.plot(original_data[:sequence_length], label='Original Data')
    plt.plot(np.arange(len(corrected_values)), corrected_values, label='Corrected Data')
    plt.legend()
    plt.title(f'Corrected Data - Magnitude: {magnitude_input}, File: {os.path.basename(file_path)}')

    # Plot both actual and corrected values
    plt.subplot(4, 1, 3)
    plt.plot(original_data, label='Actual')
    plt.plot(np.arange(len(corrected_values)), corrected_values, '--', label='Corrected Data')
    plt.legend()
    plt.title(f'Actual vs Corrected - Magnitude: {magnitude_input}, File: {os.path.basename(file_path)}')

    # Plot differences between actual and corrected values
    plt.subplot(4, 1, 4)
    plt.plot(np.arange(len(differences)), differences, label='Differences')
    plt.axhline(anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')
    #plt.scatter([index + sequence_length for index in anomalies_indices], differences[anomalies_indices], c='red', label='Anomalies')
    plt.scatter([index for index in anomalies_indices], differences[anomalies_indices], c='red', label='Anomalies')
    plt.legend()
    plt.title(f'Differences between Original and Corrected Data - Magnitude: {magnitude_input}, File: {os.path.basename(file_path)}')

    plt.tight_layout()
    plt.show()



    
def main():
    magnitude_input = input("Enter the magnitude (D, H, Z, I, F): ").upper()

    # Validate the input
    valid_magnitudes = ['D', 'H', 'Z', 'I', 'F']
    if magnitude_input not in valid_magnitudes:
        print("Invalid magnitude input. Please enter one of: D, H, Z, I, F.")
        return

    magnitude_dict = {'D': 'D(deg)', 'H': 'H(nT)', 'Z': 'Z(nT)', 'I': 'I(deg)', 'F': 'F(nT)'}
    model_dict = {'D': 'ddeg', 'H': 'hnt', 'Z': 'znt', 'I': 'ideg', 'F': 'fnt'}

    # Get column and model:
    get_column = magnitude_dict[magnitude_input]
    get_model = model_dict[magnitude_input]

    # Construct model filename based on user input
    model_filename = f'LSTM1_correctedDATA_{get_model}_v1.h5'

    # Parameters to execute:

    threshold_dict = {'D': 0.0075, 'H': 30, 'Z': 4, 'I': 0.0025, 'F': 4.5}


    scaler = MinMaxScaler()
    column_name = f'{get_column}'
    model = load_model(model_filename)
    sequence_length = 10

    print(model.summary())


    
    # Get the threshold for the current magnitude
    anomaly_threshold = threshold_dict[magnitude_input]

    #file_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM1-DETAILS\\DATA_TEST\\huan_181123.min'
    #file_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM1-DETAILS\\DATA_TEST\\jica_190417.min'
    file_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM1-DETAILS\\DATA_TEST\\jica_190814.min'


    # Use of the correct_data_with_lstm function
    corrected_values, anomalies_indices = correct_data_with_lstm(model, file_path, column_name, sequence_length, anomaly_threshold, scaler)

    # Load the single file data
    single_file_df = load_data_single_file(file_path)

    # Extract a single magnitude from the DataFrame
    original_data = single_file_df[column_name].values

    # Get the differences between Original_values and corrected_values
    differences = np.abs(np.array(original_data) - np.array(corrected_values))

    # Plot results
    plot_results(original_data, corrected_values, differences, anomalies_indices, anomaly_threshold, sequence_length, file_path, magnitude_input)



    
if __name__ == "__main__":
    main()

