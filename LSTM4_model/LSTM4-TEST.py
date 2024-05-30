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




def create_sequences(df, column_name, sequence_length, scaler):

    output_seq_length = 3

    
    # Extract 'D' (declination) column from the DataFrame
    raw_data = df[column_name].values
    #scaler = MinMaxScaler()
    
    # Reshape the 1D array to a 2D array
    raw_data_reshaped = raw_data.reshape(-1, 1)

    # Apply MinMaxScaler
    data = scaler.fit_transform(raw_data_reshaped)
    
    #print(data)

    # Initialize empty lists to store input sequences (X) and target values (y)
    X_sequences, y_targets = [], []

    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length):
        # Extract a sequence of length sequence_length as input (X)
        X_seq = data[i:i + sequence_length]
        # Extract the next value as the target (y)
        y_target = data[i + sequence_length: i + sequence_length + output_seq_length]
        
        # Check if y_target has exactly 3 elements
        if len(y_target) == output_seq_length:
            # Append the sequences to the lists
            X_sequences.append(X_seq)
            y_targets.append(y_target)
        
    # Convert lists to NumPy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    return X_sequences, y_targets




def custom_loss(y_true, y_pred):
    # Calculate the lengths of true and predicted sequences
    true_seq_length = tf.shape(y_true)[1]
    pred_seq_length = tf.shape(y_pred)[1]
    
    # Determine the minimum sequence length
    min_seq_length = tf.minimum(true_seq_length, pred_seq_length)
    
    # Compute MSE for the overlapping part of the sequences
    loss = tf.reduce_mean(tf.square(y_true[:, :min_seq_length, :] - y_pred[:, :min_seq_length, :]))
    
    return loss







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

    # Print the shapes of the concatenated sequences for verification
    #print("Concatenated X Sequences Shape:", concatenated_X_sequences.shape)
    #print("Concatenated y Targets Shape:", concatenated_y_targets.shape)

    return concatenated_X_sequences, concatenated_y_targets



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

def evaluate_single_file(model, file_path, column_name, sequence_length, anomaly_threshold, scaler):
    # Load the single file data
    single_file_df = load_data_single_file(file_path)

    # Extract a single magnitude from the DataFrame
    original_data = single_file_df[column_name].values    

    # Create sequences for the single file
    X_sequences, y_targets = create_sequences(single_file_df, column_name, sequence_length, scaler)

    # Reshape X_sequences to match the input shape expected by LSTM
    #X_sequences_reshaped = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))

    print(len(X_sequences[0]))
    print(len(y_targets[0]))

    input_sequences = X_sequences
    output_sequences = y_targets


    input_seq_length = input_sequences.shape[1]
    output_seq_length = output_sequences.shape[1]
    

    # Define input and output dimensions
    input_dim = input_sequences.shape[2]
    output_dim = output_sequences.shape[2]

    latent_dim = 64


    
    
    # Predict using the trained model
    #predictions_scaled = model.predict([input_sequences, input_sequences])

    predictions_scaled = model.predict(input_sequences)

    
    print("Predicted output shape: ", predictions_scaled.shape)

    predictions_scaled = predictions_scaled.squeeze().tolist()

    #print(y_targets)

    y_targets = y_targets.squeeze().tolist()

    #print(type(y_targets))
    #print(y_targets)

    # Inverse transform the scaled predictions
    predictions = scaler.inverse_transform(predictions_scaled)

    ##########
    y_targets = scaler.inverse_transform(y_targets)

    #print(type(input_sequences))
    #print(input_sequences)

    input_seq = input_sequences.squeeze().tolist()

    x_input = scaler.inverse_transform(input_seq)

    #print(predictions)
    #print(len(predictions))
    #print(type(predictions))
    
    #print(y_targets)
    #print(len(y_targets))
    #print(type(y_targets))

    #print(input_sequences)


    #xgr1 = range(10)
    #xgr2 = range(10,14)
    #plt.plot(xgr1, x_input)

    for i in range(len(predictions)):
        plot_predictions(x_input, y_targets, predictions, i)



    
    
    return




def plot_predictions(input_sequences, y_targets, predictions, sample_index):
    input_sequence = input_sequences[sample_index]
    y_target = y_targets[sample_index]
    prediction = predictions[sample_index]

    # Plot input sequence
    plt.plot(range(len(input_sequence)), input_sequence, label='Input Sequence', marker='o')

    # Plot actual values
    plt.plot(range(len(input_sequence), len(input_sequence) + len(y_target)), y_target, label='Actual Values', marker='o')

    # Plot predictions
    plt.plot(range(len(input_sequence), len(input_sequence) + len(prediction)), prediction, label='Predictions', marker='o')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'Sample {sample_index + 1}')
    plt.legend()
    plt.grid(True)
    plt.show()



def evaluate_folder(folder_path, sequence_length, anomaly_threshold, column_name, model, scaler):
    # Get a list of all .min files in the folder
    min_files = [f for f in os.listdir(folder_path) if f.endswith('.min')]

    # Loop through each .min file
    for min_file in min_files:
        file_path = os.path.join(folder_path, min_file)
        evaluate_single_file(model, file_path, column_name, sequence_length, anomaly_threshold, scaler)


    return

    
def main():
    '''
    #magnitude_columns = ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    #threshold = [0.002, 2.5, ]

    scaler = MinMaxScaler()
    column_name = 'H(nT)'
    model = load_model('LSTM1_jica_hnt_v1.h5')
    sequence_length = 10
    anomaly_threshold = 0.002
    folder_path = 'C:\\Users\\PRACTICAS - IGP\\Documents\\TEST4-GENERAL\\JICA_TEST'

    evaluate_folder(folder_path, sequence_length, anomaly_threshold, column_name, model, scaler)
    '''
    # Get user input for magnitude (D, H, Z, I, F)
    magnitude_input = input("Enter the magnitude (D, H, Z, I, F) (Solo se tiene archivo de D): ").upper()

    # Validate the input
    valid_magnitudes = ['D', 'H', 'Z', 'I', 'F']
    if magnitude_input not in valid_magnitudes:
        print("Invalid magnitude input. Please enter one of: D, H, Z, I, F.")
        return

    magnitude_dict = {'D': 'D(deg)', 'H': 'H(nT)', 'Z': 'Z(nT)', 'I': 'I(deg)', 'F': 'F(nT)'}
    model_dict = {'D': 'ddeg', 'H': 'hnt', 'Z': 'znt', 'I': 'ideg', 'F': 'fnt'}

    # Get column and model:
    get_column = magnitude_dict[magnitude_input]
    get_model  = model_dict[magnitude_input]
    
    # Construct model filename based on user input
    model_filename = f'LSTM4_correctedDATA_{get_model}_v1.h5'

    # Create MinMaxScaler and load model
    scaler = MinMaxScaler()
    column_name = f'{get_column}'
    model = load_model(model_filename, custom_objects={'custom_loss': custom_loss})

    print("Input shape: ", model.input_shape)
    print("Output shape: ", model.output_shape)

    # Define a dictionary to map each magnitude to its threshold
    threshold_dict = {'D': 0.25, 'H': 3.5, 'Z': 1, 'I': 0.0025, 'F': 4.5}

    # Get the threshold for the current magnitude
    anomaly_threshold = threshold_dict[magnitude_input]

    # Set other parameters
    sequence_length = 10
    folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM4-Train\\TEST_SET'

    # Call evaluate_folder with user-defined parameters
    evaluate_folder(folder_path, sequence_length, anomaly_threshold, column_name, model, scaler)

    
    
if __name__ == "__main__":
    main()

