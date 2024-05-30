import os
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



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
        #df = pd.read_table(file_path, skiprows=2, delim_whitespace=True)
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
        y_target = data[i + sequence_length]
        
        # Append the sequences to the lists
        X_sequences.append(X_seq)
        y_targets.append(y_target)

    # Convert lists to NumPy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    return X_sequences, y_targets




def train_and_save_lstm_model(data_frames, column_name, sequence_length, model_filename, scaler):
    #Empty lists to store sequences for all DataFrames
    all_X_sequences, all_y_targets = [], []

    # Iterate through each DataFrame to create sequences
    for df in data_frames:
        X_sequences, y_targets = create_sequences(df, column_name, sequence_length, scaler)
        all_X_sequences.append(X_sequences)
        all_y_targets.append(y_targets)  

    # Concatenate sequences from all DataFrames
    concatenated_X_sequences = np.concatenate(all_X_sequences)
    concatenated_y_targets = np.concatenate(all_y_targets)
    
    # Reshape X_sequences to match the input shape expected by LSTM
    X_sequences_reshaped = concatenated_X_sequences.reshape(
        (concatenated_X_sequences.shape[0], concatenated_X_sequences.shape[1], 1)
    )

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(concatenated_X_sequences.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Print a summary of the model architecture
    print(model.summary())

    # Train the model
    model.fit(X_sequences_reshaped, concatenated_y_targets, epochs=10, batch_size=32, validation_split=0.2)

    # Save the trained model
    model.save(model_filename)
    print(f"Model saved to {model_filename}")

    return 



def main():

    # List of magnitude column names
    magnitude_columns = ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']


    # Mapping a dictionary to map variations to standarized names
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
    

    # Training!
    folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM1-TEST\\Corrected2010_2020(decompressed)\\complete_data'

    loaded_data = load_data(folder_path)
    data_frames = loaded_data
    sequence_length = 10
    scaler = MinMaxScaler()

    # Apply the mapping to your loaded data
    for df in data_frames:
        # Rename columns based on the mapping
        df.rename(columns=column_mapping, inplace=True)


    for column_name in magnitude_columns:
        model_filename = f'LSTM1_correctedDATA_{column_name.replace("(", "").replace(")", "").replace("/", "_").replace(" ", "_").lower()}_v1.h5'
        train_and_save_lstm_model(data_frames, column_name, sequence_length, model_filename, scaler)
    

    
if __name__ == "__main__":
    main()

