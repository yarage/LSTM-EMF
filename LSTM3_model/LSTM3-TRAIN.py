import os
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model  # Added Model import
from tensorflow.keras.layers import LSTM, Dense, Input  # Added Input import
from tensorflow.keras.layers import Attention  # Added Attention import
#from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import random




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
    X_sequences, y_targets = [], []

    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length + 1):
        # Extract sequences of length sequence_length as input (X)
        X_seq = data[i:i + sequence_length]
        # Extract the next value as the target (y)
        #print(X_seq)

    
        # Define the range for selecting the central point
        central_point_range = (len(X_seq) // 2) - 3, (len(X_seq) // 2) + 3
        
        # Select a random central point
        central_point = random.randint(central_point_range[0], central_point_range[1])
        
        # Get the first value of the curve
        first_value = X_seq[0]


        min_change = []
        max_change = []

        # Calculate the range for the new value (1% to 25% of the first value)
        for i in range(len(columns)):
            min_change.append(0.10 * first_value[i])
            max_change.append(0.35 * first_value[i])

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

def train_and_save_lstm_model(data_frames, columns, sequence_length, model_filename, scaler):
    # Empty lists to store sequences for all DataFrames
    all_X_sequences, all_y_targets = [], []

    # Iterate through each DataFrame to create sequences
    for df in data_frames:
        X_sequences, y_targets = create_sequences(df, columns, sequence_length, scaler)
        all_X_sequences.append(X_sequences)
        all_y_targets.append(y_targets)


    print("#######################")
    print(all_X_sequences[0][0])
    print(len(all_X_sequences[0][0]))
    print(type(all_X_sequences))
    print(len(all_X_sequences))
    

    print("#######################")    
    print(all_y_targets[0][0])
    print(len(all_y_targets[0][0]))
    print(type(all_y_targets))
    print(len(all_y_targets))
    


    # Concatenate sequences from all DataFrames
    concatenated_X_sequences = np.concatenate(all_X_sequences)
    concatenated_y_targets = np.concatenate(all_y_targets)

    input_sequences = concatenated_X_sequences
    output_sequences = concatenated_y_targets

    input_seq_length = input_sequences.shape[1]

    # Define input and output dimensions
    input_dim = input_sequences.shape[2]
    output_dim = output_sequences.shape[2]

    latent_dim = 64

    # Define the encoder
    encoder_inputs = Input(shape=(input_seq_length, input_dim))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Define the decoder
    decoder_inputs = RepeatVector(1)(encoder_outputs)  # Predict the next single value
    decoder_lstm = LSTM(latent_dim, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs)
    decoder_dense = TimeDistributed(Dense(output_dim, activation='linear'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model(encoder_inputs, decoder_outputs)

    # Print a summary of the model architecture
    print(model.summary())

    # Compile the model with custom loss function
    model.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error loss

    # Parameters to train
    batch_size = 32
    epochs = 10
    validation_split = 0.2



    # Train the model
    model.fit(input_sequences, output_sequences,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split)

    # Save the trained model
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    
    return

def main():
    # List of magnitude column names
    columns = ['D(deg)', 'H(nT)', 'Z(nT)']

    # Training data folder path
    folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM3-Train\\TRAIN_SET'

    # Load data
    loaded_data = load_data(folder_path)
    data_frames = loaded_data

    # Mapping a dictionary to map variations to standardized names
    column_mapping = {
        'D(Deg)': 'D(deg)',
        'D': 'D(deg)',
        'D(deg)': 'D(deg)',
        'H(nT)': 'H(nT)',
        'H': 'H(nT)',
        'Z(nT)': 'Z(nT)',
        'Z': 'Z(nT)',
        'I(Deg)': 'I(deg)',
        'I': 'I(deg)',
        'I(deg)': 'I(deg)',
        'F(nT)': 'F(nT)',
        'F': 'F(nT)'
    }

    # Apply the mapping to your loaded data
    for df in data_frames:
        # Rename columns based on the mapping
        df.rename(columns=column_mapping, inplace=True)

    # Sequence length indicates how many values are considered to make a prediction
    sequence_length = 10
    scaler = MinMaxScaler()

    # Train a single model for all magnitudes
    model_filename = 'LSTM3_offset_model.h5'
    train_and_save_lstm_model(data_frames, columns, sequence_length, model_filename, scaler)


if __name__ == "__main__":
    main()
