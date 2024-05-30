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
    
    # Print the shapes of the concatenated sequences, Optional
    #print("Concatenated X Sequences Shape:", concatenated_X_sequences.shape)
    #print("Concatenated y Targets Shape:", concatenated_y_targets.shape)

    input_sequences = concatenated_X_sequences
    output_sequences = concatenated_y_targets


    input_seq_length = input_sequences.shape[1]
    output_seq_length = output_sequences.shape[1]


    print(input_sequences.shape[0])
    print(output_sequences.shape[0])
    
    print(input_sequences.shape[1])
    print(output_sequences.shape[1])



    # Define input and output dimensions
    input_dim = input_sequences.shape[2]
    output_dim = output_sequences.shape[2]


    print(input_sequences.shape[2])
    print(output_sequences.shape[2])


    latent_dim = 64

    '''
    # Define the encoder
    encoder_inputs = Input(shape=(None, input_dim))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = Input(shape=(None, output_dim))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='linear')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    '''

    
    # Define the encoder
    encoder_inputs = Input(shape=(input_seq_length, input_dim))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = RepeatVector(output_seq_length)(encoder_outputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs)
    decoder_dense = TimeDistributed(Dense(output_dim, activation='linear'))
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model
    model = Model(encoder_inputs, decoder_outputs)
                
    

    # Print a summary of the model architecture
    print(model.summary())

    # Compile the model with custom loss function
    model.compile(optimizer='adam', loss=custom_loss)


    # Parameters to train
    batch_size = 32
    epochs = 10
    validation_split = 0.2

    '''
    # Train the model
    model.fit([input_sequences, input_sequences], output_sequences,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split)
    '''

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

    # Training! - Better way:
    folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM4-Train\\TRAIN_SET'
    #C:\Users\Practicas-IGP\Documents\LSTM1-TEST\Corrected2010_2020(decompressed)\complete_data
    loaded_data = load_data(folder_path)
    data_frames = loaded_data

    # sequence_length indicates how many values are considered to make a prediction
    sequence_length = 10
    scaler = MinMaxScaler()



    # Apply the mapping to your loaded data
    for df in data_frames:
        # Rename columns based on the mapping
        df.rename(columns=column_mapping, inplace=True)
    

    # Generate some dummy data
    #num_samples = 1000
    #input_seq_length = 10
    #output_seq_length = 3  # Changed to 3
    

    #X = np.random.randn(num_samples, input_seq_length, input_dim)
    #Y = np.random.randn(num_samples, output_seq_length, output_dim)

    for column_name in magnitude_columns:
            model_filename = f'LSTM1_correctedDATA_{column_name.replace("(", "").replace(")", "").replace("/", "_").replace(" ", "_").lower()}_v1.h5'
            train_and_save_lstm_model(data_frames, column_name, sequence_length, model_filename, scaler)
        

   
    


    
if __name__ == "__main__":
    main()

