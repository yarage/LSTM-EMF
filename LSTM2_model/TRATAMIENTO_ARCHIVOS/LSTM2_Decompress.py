import os
import gzip
import shutil

def decompress_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .min.gz files in the input folder
    compressed_files = [file for file in os.listdir(input_folder) if file.endswith('.min.gz')]

    for compressed_file in compressed_files:
        input_path = os.path.join(input_folder, compressed_file)
        output_path = os.path.join(output_folder, compressed_file.replace('.gz', ''))

        try:
            with gzip.open(input_path, 'rt') as compressed_file:
                # Read the decompressed data
                data = compressed_file.read()

                # Write the decompressed data to a new file
                with open(output_path, 'w') as output_file:
                    output_file.write(data)

        except (EOFError, gzip.BadGzipFile):
            print(f"Warning: Empty or corrupted file found: {input_path}")
            continue

    print("Decompression completed.")

# Specify the input and output folders
input_folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TEST_25(6)'
output_folder_path = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\DECO_TEST_25(6)'


#C:\Users\Practicas-IGP\Documents\LSTM-CORRECTION\DATA_1enero2000-actualidad



# Call the decompression function
decompress_files(input_folder_path, output_folder_path)
