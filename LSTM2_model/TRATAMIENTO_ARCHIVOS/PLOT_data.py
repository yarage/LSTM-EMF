import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_magnitudes(file_path):
    # Extract the filename to show it in the plot
    filename = os.path.basename(file_path).split('.')[0]

    # Read the data from the file
    column_names = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    data = pd.read_csv(file_path, sep='\s+', skiprows=3, names=column_names)

    # Create a continuous time axis covering the entire day (1440 minutes)
    complete_time_range = pd.date_range(start=f"{data['YYYY'].iloc[0]}-{data['MM'].iloc[0]}-{data['DD'].iloc[0]} 00:00",
                                        end=f"{data['YYYY'].iloc[-1]}-{data['MM'].iloc[-1]}-{data['DD'].iloc[-1]} 23:59",
                                        freq='min')

    # Create a DataFrame with NaN values for all magnitudes
    zero_data = pd.DataFrame(index=complete_time_range, columns=['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)'])

    # Insert the actual data into the DataFrame based on the timestamps
    data['timestamp'] = pd.to_datetime(data[['YYYY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H %M')
    zero_data.loc[data['timestamp'], ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']] = data[['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']].values

    # Plot each magnitude on a separate subplot
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    for i, magnitude in enumerate(magnitudes):
        axs[i].plot(zero_data.index, zero_data[magnitude], label=magnitude)
        axs[i].set_ylabel(magnitude)

    # Set common labels and title
    axs[-1].set_xlabel('Time')
    plt.suptitle(f'Evolution of Magnitudes Over Time - {filename}')

    # Display legend
    plt.legend()

    # Display the plot
    plt.show()
    

def plot_magnitudes_in_folder(folder_path):
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .min file
        if file_name.endswith('.min'):
            # Get the full file path
            file_path = os.path.join(folder_path, file_name)
            # Call the plot_magnitudes function for the current file
            plot_magnitudes(file_path)


def plot_month_data(folder_path, station_code, year, month):
    # Determine the number of days in the month
    complete_year = 2000 + year if year < 100 else year  # Handle 2-digit year format
    days_in_month = pd.Timestamp(complete_year, month, 1).days_in_month

    # Create a date range for the entire month
    date_range = pd.date_range(start=f'{complete_year}-{month:02d}-01', end=f'{complete_year}-{month:02d}-{days_in_month} 23:59:00', freq='min')

    # Create a DataFrame with NaN values for the entire month
    column_names = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    nan_data = pd.DataFrame(np.nan, index=date_range, columns=column_names)

    # Get a list of all .min files in the folder
    min_files = [f for f in os.listdir(folder_path) if f.endswith('.min') and f.startswith(f'{station_code}_{year}{month:02}')]

    # Iterate through each file
    for min_file in min_files:
        file_path = os.path.join(folder_path, min_file)

        # Read the data from the file
        column_names = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
        data = pd.read_csv(file_path, sep = '\s+', skiprows=3, names=column_names)

        # Convert the date and time columns to datetime format
        data['datetime'] = pd.to_datetime(data[['YYYY', 'MM', 'DD', 'hh', 'mm']].astype(str).agg(' '.join, axis=1), format='%Y %m %d %H %M')

        # Filter out rows with specific values
        data = data.replace({99.9999: np.nan, 99999.9: np.nan})

        # Update the nan_data DataFrame with measured data
        nan_data.loc[data['datetime'], ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']] = data[['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']].values

    # Create subplots
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    # Plot each magnitude on a separate subplot with labels and colors
    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, magnitude in enumerate(magnitudes):
        axs[i].plot(nan_data.index, nan_data[magnitude], label=f'{magnitude}', color=colors[i])
        #axs[i].legend()
        axs[i].set_ylabel(f'{magnitude}')
    
    # Set common labels and title
    axs[-1].set_xlabel('Time')
    plt.suptitle(f'Evolution of Magnitudes Over Time - Station: {station_code}, Year: {complete_year}, Month: {month:02}')

    # Set the X-axis limits to show an entire month
    plt.xlim(nan_data.index[0], nan_data.index[-1])

    plt.show()


def plot_year_data(folder_path, station_code, year):
    # Iterate over each month of the year
    for month in range(1, 13):
        # Plot data for the current month
        plot_month_data(folder_path, station_code, year, month)


# USAGE:


file_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TESTING_SET\\huan_151116.min'

#folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TESTING_SET'
#folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\DECO_DATA_1enero2000-actualidad'
#folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TEST_SET(complete)'
#folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TEST_SET'
folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\DECO_TEST_25(6)'


station_code = 'areq'
year = 19

#plot_year_data(folder_path, station_code, year)
#plot_magnitudes(file_path)
plot_magnitudes_in_folder(folder_path)






