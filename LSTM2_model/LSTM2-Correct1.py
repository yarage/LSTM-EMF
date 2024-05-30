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

import matplotlib.pyplot as plt

import datetime

import itertools




############################################################################################
##### Para crear archivos .min que contengan la data corregida #############################
############################################################################################

# Las funciones siguientes, se encargan de crear archivos .min para almacenar data corregida
# Con la estructura estandar, y completando la falta de datos con 99999
def generate_min_file(station, year, month, day, data_type, input_folder, output_folder):
    # Convertir año, mes y dia a valores enteros
    year = int(year)
    month = int(month)
    day = int(day)

    # Construir el objeto datetime
    date_obj = datetime.datetime(year, month, day)

    # Construir la ruta de los archivos input y output
    input_file_name = f"{station}_{date_obj.strftime('%y%m%d')}.min"
    input_file_path = os.path.join(input_folder, input_file_name)
    output_file_name = f"{station}_{date_obj.strftime('%y%m%d')}.min"
    output_file_path = os.path.join(output_folder, output_file_name)

    # Abrir el archivo input .min para lectura
    with open(input_file_path, 'r') as input_file:
        # Leer el encabezado del archivo original
        header = input_file.readline().strip()

    # Abrir el archivo output .min para escritura
    with open(output_file_path, 'w') as output_file:
        # Copiar el encabezado original en el archivo nuevo
        output_file.write(header + '\n\n')

        # Escribiendo nombres de las columnas estandar
        output_file.write(" DD MM YYYY hh mm   D(deg)    H(nT)    Z(nT)   I(deg)    F(nT)\n\n")
        
        # Rellenando la copia del archivo min con 99999
        for hour in range(24):
            for minute in range(60):
                output_file.write(f" {date_obj.strftime('%d')} {date_obj.strftime('%m')} {date_obj.strftime('%Y')} "
                                  f"{hour:02d} {minute:02d}  99.9999  99999.9  99999.9  99.9999  99999.9\n")

    print(f" Archivo '{output_file_name}' generado exitosamente.")



def load_data_initial(folder_path):
    standard_columns = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    read_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.min')]

    data_frames = []
    for file_path in read_files:
        station_code, year, month, day = extract_info_from_filename_day(os.path.basename(file_path))
        for separator in ['\s+', '\s{2,}']:  # Por si los espacios entre columnas son diferentes
            try:
                df = pd.read_table(file_path, skiprows=2, sep=separator)
                break  # Parar si es exitoso
            except Exception as e:
                print(f"Error al leer el archivo {file_path} con separador '{separator}': {e}")
                continue

        if not df.empty:
            # Mantener las primeras 10 columnas y descartar las adicionales
            df = df.iloc[:, :10]

            # Renombrando columnas con los nombres de columnas estandar
            df.columns = standard_columns

            # Formato de la fecha, hora, y minuto
            df['MM'] = df['MM'].astype(str).str.zfill(2)
            df['DD'] = df['DD'].astype(str).str.zfill(2)
            df['hh'] = df['hh'].astype(str).str.zfill(2)
            df['mm'] = df['mm'].astype(str).str.zfill(2)

            df['Station'] = station_code
            df['Year'] = year
            df['Month'] = month

            data_frames.append((df, os.path.basename(file_path)))

    return data_frames



def copy_data_from_a_to_b(df_A, df_B):
    # Iterar sobre las filas de df_A
    for index, row in df_A.iterrows():
        # Extraer la hora y minuto de la fila
        hour = row['hh']
        minute = row['mm']

        matching_rows_indices = df_B[(df_B['hh'] == hour) & (df_B['mm'] == minute)].index
        
        if not matching_rows_indices.empty:
            # Copiar la data de df_A a df_B
            for idx in matching_rows_indices:
                df_B.loc[idx] = row
    return df_B



def copy_data_to_min_file(df, min_file_path):
    # Definir las columnasa copiar del DataFrame
    columns_to_copy = ['DD', 'MM', 'YYYY', 'hh', 'mm', 'D(deg)', 'H(nT)', 'Z(nT)', 'I(deg)', 'F(nT)']
    
    # Abrir el existente archivo .min para lectura
    with open(min_file_path, 'r') as f:
        # Leer las cinco primeras columnas y almacenarlas en el encabezado
        header = [next(f) for _ in range(4)]
        
        # No procesando el encabezado
        for _ in range(5):
            next(f)
        
        f.seek(0)
        
        # Abrir el archivo para escritura (modo sobreescribir)
        with open(min_file_path, 'w') as new_f:
            # Escribir el encabezado en el nueo archivo
            for line in header:
                new_f.write(line)
            
            # Iterar sobre cada fila del DataFrame
            for index, row in df.iterrows():
                # Extraer la data requerida de las filas del DataFrame
                data_values = [str(row[col]) for col in columns_to_copy]

                # Para dar el formato de los datos de fecha
                data_values[1:5] = [val.zfill(2) for val in data_values[1:5]]

                # Dar el formato de los datos para la precision deseada
                data_values[5] = f"{float(data_values[5]):.4f}"
                data_values[6] = f"{float(data_values[6]):.1f}"
                data_values[7] = f"{float(data_values[7]):.1f}"
                data_values[8] = f"{float(data_values[8]):.4f}"
                data_values[9] = f"{float(data_values[9]):.1f}"

                data_line = ' ' + ' '.join(data_values[:5]) + '  ' + '  '.join(data_values[5:10])

                # Escribiendo la data
                new_f.write(data_line + '\n')


def process_min_files_in_folder(input_folder, output_folder):
    # Crear la carpeta de output si es que no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterar sobre todos los archivos .min de la carpeta input
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.min'):
            # Extraer el nombre de la estacion, año, mes, dia
            parts = file_name.split('_')
            station = parts[0]
            year = int(parts[1][:2]) + 2000
            month = int(parts[1][2:4])
            day = int(parts[1][4:6])

            # Usar la funcion generate_min_file con la informacion extraida
            generate_min_file(station, year, month, day, 'Completed', input_folder, output_folder)




def complete_folder9(folder_a, folder_b):
    # Para crear una copia de los archivos de un folder a otro,
    # pero sin datos (Solo estructura)
    process_min_files_in_folder(folder_a, folder_b)
    df_A = load_data_initial(folder_a)
    df_B = load_data_initial(folder_b)
    df_completed = []
    for i in range(len(df_A)):
        df_aux = copy_data_from_a_to_b(df_A[i][0], df_B[i][0])
        df_completed.append(df_aux)
        file_name = df_B[i][1]
        min_file_path = os.path.join(folder_b, file_name)
        copy_data_to_min_file(df_completed[i], min_file_path)
        print(f" Archivo copia y sin datos '{file_name}' creado exitosamente.")


############################################################################################
############################################################################################
############################################################################################


def extract_info_from_filename_day(file_name):
    match = re.match(r'([a-zA-Z]+)_(\d{6})\.min', file_name)
    if match:
        station_code, date_str = match.groups()
        year = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:])
        return station_code, year, month, day
    else:
        raise ValueError(f"Formato Invalido Archivo: {file_name}")


def extract_info_from_filename(file_name):
    match = re.match(r'([a-zA-Z]+)_(\d{6})\.min', file_name)
    if match:
        station_code, date_str = match.groups()
        year = int(date_str[:2])
        month = int(date_str[2:4])
        return station_code, year, month
    else:
        raise ValueError(f"Formato Invalido Archivo: {file_name}")


def create_sequences(df, sequence_length, scaler):
    output_seq_length = 1  # Prediciendo el siguiente valor para cada columna, definido por el modelo (valor fijo)

    raw_data = df
    
    # Aplicando MinMaxScaler (Escalando los valores para mejorar su tratamiento)
    data = scaler.fit_transform(raw_data)

    X_sequences, y_targets = [], []

    # Iterando en la data para crear secuencias
    for i in range(len(data) - sequence_length - output_seq_length + 1):
        X_seq = data[i:i + sequence_length]
        y_target = data[i + sequence_length:i + sequence_length + output_seq_length]

        X_sequences.append(X_seq)
        y_targets.append(y_target)

    # De listas a NumPy arrays
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)

    return X_sequences, y_targets




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
        # Verificando si existen las columnas 
        columns_to_drop = ['DD', 'MM', 'YYYY', 'hh', 'mm']
        existing_columns = set(df.columns)
        columns_to_drop = [col for col in columns_to_drop if col in existing_columns]

        # Descartando las columnas no deseadas
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Renombrando las columnas de acuerdo al diccionario
        df.rename(columns=column_mapping, inplace=True)

        df['Station'] = station_code
        df['Year'] = year
        df['Month'] = month

    return df


def evaluate_whole_df(model, single_file_df, sequence_length, scaler, columns):
    # Creando las secuencias para un solo archivo
    X_sequences, y_targets = create_sequences(single_file_df, sequence_length, scaler)
    input_sequences = X_sequences
    output_sequences = y_targets

    # Uso del modelo
    predictions_scaled = model.predict(input_sequences)

    # Convirtiendo a listas    
    predictions_scaled = predictions_scaled.squeeze().tolist()
    y_targets = y_targets.squeeze().tolist()

    # Las predicciones estan escaladas (entre 0 y 1), asi que se invierte el proceso para tener los valores correctos
    predictions = scaler.inverse_transform(predictions_scaled)
    y_targets = scaler.inverse_transform(y_targets)
    x_inic = scaler.inverse_transform(input_sequences[0])

    pred_D = list(x_inic[:, 0]) + list(predictions[:, 0])
    pred_H = list(x_inic[:, 1]) + list(predictions[:, 1])
    pred_Z = list(x_inic[:, 2]) + list(predictions[:, 2])
    
    return pred_D, pred_H, pred_Z


def plot_subplots_DHZ(list1, list2, list3, original_data, filename, prepost, cond):

    D_orig = list(original_data[:, 0])
    H_orig = list(original_data[:, 1])
    Z_orig = list(original_data[:, 2])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)']

    num_minutes = len(list1)  
    # Para que el eje x sean los minutos en el dia
    minutes_in_day = np.arange(0, num_minutes, 60)  
    hours = minutes_in_day // 60  
    minutes = minutes_in_day % 60 
    x_labels = [f'{h:02d}:{m:02d}' for h, m in zip(hours, minutes)] 

    
    if cond == 0:
        axes[0].plot(D_orig, label='D(deg) - actual', color='blue')
        axes[1].plot(H_orig, label='H(nT)- actual', color='green')
        axes[2].plot(Z_orig, label='Z(nT)- actual', color='red')
        
    if cond == 1:

        axes[0].plot(list1, label='D(deg) - Modelo',  color='blue')
        axes[1].plot(list2, label='H(nT) - Modelo', color='green')
        axes[2].plot(list3, label='Z(nT) - Modelo', color='red')
        axes[0].plot(D_orig, '--', label='D(deg) - reportado', color='orange')
        axes[1].plot(H_orig, '--', label='H(nT)- reportado', color='orange')
        axes[2].plot(Z_orig, '--', label='Z(nT)- reportado', color='orange')
    
    plt.xticks(minutes_in_day, x_labels, rotation=45)
    
    for i, magnitude in enumerate(magnitudes):
        axes[i].set_ylabel(magnitude)
        axes[i].legend()

    axes[-1].set_xlabel('Time UTC')
    plt.suptitle(prepost + " Archivo: {}".format(filename))
    
    plt.tight_layout()
    plt.show()



def detection_anomalies(list1, list2, list3, original_data, tolerance):
    D_orig = np.array(original_data[:, 0])
    H_orig = np.array(original_data[:, 1])
    Z_orig = np.array(original_data[:, 2])

    list1 = np.array(list1)
    list2 = np.array(list2)
    list3 = np.array(list3)

    D_dif = np.abs(D_orig - list1)
    H_dif = np.abs(H_orig - list2)
    Z_dif = np.abs(Z_orig - list3)


    min_D_orig = np.nanmin(D_orig)
    max_D_orig = np.nanmax(D_orig)
    thr_D_orig = abs(max_D_orig - min_D_orig) * tolerance / 100
    

    min_H_orig = np.nanmin(H_orig)
    max_H_orig = np.nanmax(H_orig)
    thr_H_orig = abs(max_H_orig - min_H_orig) * tolerance / 100
    
    min_Z_orig = np.nanmin(Z_orig)
    max_Z_orig = np.nanmax(Z_orig)
    thr_Z_orig = abs(max_Z_orig - min_Z_orig) * tolerance / 100
    
    anomaly_threshold = [thr_D_orig, thr_H_orig, thr_Z_orig]

    magnitudes = ['D(deg)', 'H(nT)', 'Z(nT)']
    
    D_anomalies = np.where(D_dif > thr_D_orig)[0]
    H_anomalies = np.where(H_dif > thr_H_orig)[0]
    Z_anomalies = np.where(Z_dif > thr_Z_orig)[0]

    # Encontrando los indices de los valores nan del array original
    nan_indices_1 = np.where(np.isnan(D_orig))[0]
    nan_indices_2 = np.where(np.isnan(H_orig))[0]
    nan_indices_3 = np.where(np.isnan(Z_orig))[0]

    # Concatenando los arrays
    D_anomalies_nan = np.concatenate((nan_indices_1, D_anomalies))
    H_anomalies_nan = np.concatenate((nan_indices_2, H_anomalies))
    Z_anomalies_nan = np.concatenate((nan_indices_3, Z_anomalies))

    D_anomalies_nan = np.sort(D_anomalies_nan)
    H_anomalies_nan = np.sort(H_anomalies_nan)
    Z_anomalies_nan = np.sort(Z_anomalies_nan)

    print("Posición de anomalías D:")
    print(D_anomalies_nan)
    
    print("Posición de anomalías H:")
    print(H_anomalies_nan)
    
    print("Posición de anomalías Z:")
    print(Z_anomalies_nan)
    
    return D_anomalies_nan, H_anomalies_nan, Z_anomalies_nan, anomaly_threshold

def compare_arrays_DHZ(A, B):
    differences_D = []
    differences_H = []
    differences_Z = []

    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != B[i][j]:
                if j == 0:
                    differences_D.append(i)    
                if j == 1:
                    differences_H.append(i)    
                if j == 2:
                    differences_Z.append(i)    

    return differences_D, differences_H, differences_Z


############################################################################################
####### Funciones para detectar valores consecutivos en una lista y agruparlos #############
############################################################################################

def consecutive_groups(lst):
    groups = []
    current_group = []
    for i in range(len(lst)):
        if i == 0 or lst[i] != lst[i - 1] + 1:
            if current_group:
                groups.append(current_group)
            current_group = [lst[i]]
        else:
            current_group.append(lst[i])
    if current_group:
        groups.append(current_group)
    return groups

def analyze_consecutive_values(*lists):
    results = []
    for lst in lists:
        groups = consecutive_groups(lst)
        group_info = []
        for group in groups:
            group_info.append([
                len(group),
                min(group),
                max(group)
            ])
        results.append(group_info)
    return results

############################################################################################
############################################################################################
############################################################################################

def identify_punctual_errors(single_file_df, anomaly_threshold, model, scaler, sequence_length, max_consecutives, pos_groups_anomalies):

    corrected_data_part = np.copy(single_file_df)
    X_sequences, y_targets = create_sequences(single_file_df, sequence_length, scaler)

    for ct in pos_groups_anomalies:
        min_value = np.min(ct)
        max_value = np.max(ct)

        '''
        print("Min value: ", min_value)
        print("Max value: ", max_value)

        print("TEST:", range(min_value - sequence_length - 3, max_value))
        print("X_sequences len: ", len(X_sequences))
        
        print("threshold")
        print(anomaly_threshold)
        '''
        
        v_initial = 0

        lim_inf = min_value - sequence_length - 3
        if lim_inf < 0:
            lim_inf = 0
        lim_sup = max_value - sequence_length + 3
        if lim_sup - len(X_sequences) > 0:
            lim_sup = len(X_sequences)
        for i in range(lim_inf, lim_sup):
            input_seq = X_sequences[i].reshape(1, sequence_length, 3)
            output_seq = y_targets[i].reshape(1, 1, 3)

            predictions_scaled = model.predict(input_seq)
            predictions_i = scaler.inverse_transform(predictions_scaled[0])
            y_targets_i = scaler.inverse_transform(output_seq[0])

            diff_i = np.abs(y_targets_i - predictions_i)

            flag_change = 0
            for j in range(3):
                if diff_i[0][j] > anomaly_threshold[j] or np.isnan(y_targets_i[0][j]) == True:

                    # Aqui ocurre el cambio del valor por el predicho por el modelo
                    # Podría cambiarse para que se obtenga el valor por interpolacion, si se deseara
                    # Incluso podría cambiarse para que no cambie el valor y simplemente lo elimine asignandole
                    # el valor de np.nan
                    corrected_data_part[i + sequence_length][j] = predictions_i[0][j]
                    
                    #OJO:
                    print("ANTES: ", single_file_df[i + sequence_length][j])
                    single_file_df[i + sequence_length][j] = predictions_i[0][j]
                    
                    print("DESPUES: ", single_file_df[i + sequence_length][j])
                    flag_change += 1

                if flag_change > 0:
                    X_sequences, y_targets = create_sequences(single_file_df, sequence_length, scaler)
        
    differences_D, differences_H, differences_Z = compare_arrays_DHZ(single_file_df, corrected_data_part)

    print("Valores:", differences_D, differences_H, differences_Z)
    
    results = analyze_consecutive_values(differences_D, differences_H, differences_Z)

    for i in range(3):
        for j in results[i]:
            if j[0] > max_consecutives:
                vi = j[1]
                vf = j[2] + 1
                for k in range(vi,vf):
                    corrected_data_part[k][i] = single_file_df[k][i]

    return corrected_data_part



def corrected_data(model, single_file_df, sequence_length, anomaly_threshold, scaler, tolerance, max_consecutives, positions_anomalies):

    # Positions_anomalies contienen las posiciones de anomalias de cada columna (D, H, Z)
    # por ello juntamos estas posiciones y obtendremos las anomalias agrupadas para una ejecucion mas eficiente
    pos_anomalies = np.concatenate(positions_anomalies)
    pos_anomalies = np.unique(pos_anomalies)
    pos_groups_anomalies = []
    sublist = []
    
    for i in range(len(pos_anomalies)):
        sublist.append(pos_anomalies[i])
        
        # Verificamos si la diferencia entre el elemento actual y el siguiente es mayor a 10 para dividir en otro grupo
        # Hacemos ello con el fin de solo realizar una correccion y analizar los minutos donde haya anomalias
        # Se toma la diferencia de 10 como un valor referencial, podria ser mayor o menor, considerando que si el valor
        # es muy grande ello podria hacer que la ejecucion sea lenta teniendo que evaluar y comparar más puntos de los necesarios
        if i < len(pos_anomalies) - 1 and pos_anomalies[i + 1] - pos_anomalies[i] > 10:
            pos_groups_anomalies.append(sublist)
            sublist = []
    if sublist:
        pos_groups_anomalies.append(sublist)

    print("Posiciones de Anomalias Agrupadas: ", pos_groups_anomalies)

  
    # Para corregir los errores puntuales
    corrected_data_part = identify_punctual_errors(single_file_df, anomaly_threshold, model, scaler, sequence_length, max_consecutives, pos_groups_anomalies)

    corr_D = list(corrected_data_part[:, 0])
    corr_H = list(corrected_data_part[:, 1])
    corr_Z = list(corrected_data_part[:, 2])
    
    return corr_D, corr_H, corr_Z





def create_nan_lists(A):
    nan_lists = []

    # Iterando sobre cada sublista de A
    for sublist in A:
        # Creamos una lista con valores nan de longitud especifica dada por cada sublista
        nan_list = [np.nan] * sublist[0]
        nan_lists.append(nan_list)
    return nan_lists


def evaluate_folder(folder_path, sequence_length, columns, model, scaler, tolerance):

    # Cantidad maxima de datos consecutivos a corregir
    max_consecutives = 3
    
    # Lista de todos los archivos .min en la carpeta
    min_files = [f for f in os.listdir(folder_path) if f.endswith('.min')]

    # Procesando archivo por archivo
    for min_file in min_files:
        file_path = os.path.join(folder_path, min_file)

        print("\nx==========================================x")
        print(f"    Archivo A Procesar: {min_file}    ")
        print("x==========================================x")

        # Cargando los datos del archivo:
        single_file_df = load_data_single_file(file_path)

        # Para ver los datos del archivo en pantalla:
        #print(single_file_df)

        # Assuming 'Station' contains the station code
        filename = single_file_df['Station'].iloc[0]  

        # Extraer las magnitudes del DataFrame
        original_data = single_file_df[columns].values
        
        # Copia del array original
        original_data_copy = np.copy(original_data)

        # Reemplazamos los valores sin datos por np.nan (Solo para procesar y graficar)
        original_data_copy[original_data_copy[:, 0] == 99.9999, 0] = np.nan
        original_data_copy[original_data_copy[:, 1] == 99999.9, 1] = np.nan
        original_data_copy[original_data_copy[:, 2] == 99999.9, 2] = np.nan

        original_data_copy2 = np.copy(original_data_copy)

        # Encontrar los indices de los valores NaN en cada columna
        nan_indices_column1 = np.where(np.isnan(original_data_copy[:, 0]))[0]
        nan_indices_column2 = np.where(np.isnan(original_data_copy[:, 1]))[0]
        nan_indices_column3 = np.where(np.isnan(original_data_copy[:, 2]))[0]

  
        noneData_consecutive = analyze_consecutive_values(nan_indices_column1, nan_indices_column2, nan_indices_column3)
        # Elementos de noneData_consecutive deben ser los mismos (Misma data faltante en las 3 componentes)
        # Por ello solo usaremos noneData_consecutive[0]
        # Filtramos los grupos de datos faltantes que sean de menos de 3 elementos ()

        noneData_consecutive_filtered = [] 

        for i1 in noneData_consecutive[0]:
            if i1[0] > max_consecutives:
                noneData_consecutive_filtered.append(i1)

        print("Posiciones donde faltan datos")    
        print(noneData_consecutive[0])
        print("Posiciones donde faltan datos - filtradas: ")
        print(noneData_consecutive_filtered) 



        # Para agrupar partes con datos y sin datos:
        
        original_data_filtered_none = []
        id_aux = 0

        aux_intervals =  range(len(noneData_consecutive_filtered) + 1)

        df = []

        if len(noneData_consecutive_filtered) == 0:
            df = original_data_copy
            original_data_filtered_none.append(df)
        else:    
            for i in aux_intervals:
                if i == 0:
                    df = original_data_copy[0:noneData_consecutive_filtered[i][1]]
                    if len(df) == 0:
                        print("No hay data al inicio (primeros minutos sin data) ")                    
                    else:
                        original_data_filtered_none.append(df)
                    last_lim = noneData_consecutive_filtered[i][2] + 1
                    #print("IF")

                elif i >= 0 and i < len(aux_intervals) - 1:
                    df = original_data_copy[last_lim:noneData_consecutive_filtered[i][1]]
                    original_data_filtered_none.append(df)
                    last_lim = noneData_consecutive_filtered[i][2] + 1
                    #print("ELIF")

                else:
                    df = original_data_copy[last_lim:len(original_data) + 1]
                    original_data_filtered_none.append(df)
                    #print("ELSE")

        # Para evaluar y hacer un primer uso del modelo LSTM2

        pred_D, pred_H, pred_Z = [], [], []
        for i in original_data_filtered_none:
            if len(i) >= 11:
                predi_D, predi_H, predi_Z = evaluate_whole_df(model, i, sequence_length, scaler, columns)
            else:
                predi_D, predi_H, predi_Z = [], [], []
                for k in i:
                    predi_D.append(k[0])
                    predi_H.append(k[1])
                    predi_Z.append(k[2])
                    
            pred_D.append(predi_D)
            pred_H.append(predi_H)
            pred_Z.append(predi_Z)

        pred_D_sub_df = [sublist[:] for sublist in pred_D]
        pred_H_sub_df = [sublist[:] for sublist in pred_H]
        pred_Z_sub_df = [sublist[:] for sublist in pred_Z]

        # itertools no necesita instalacion adicional
        # Obtenemos asi la data corregida:

        if len(noneData_consecutive_filtered) != 0:
            nan_lists = create_nan_lists(noneData_consecutive_filtered)

            if noneData_consecutive_filtered[0][1] == 0:
                pred_D_complete = []
                pred_H_complete = []
                pred_Z_complete = []
                for pair in itertools.zip_longest(nan_lists, pred_D):
                    for val in pair:
                        if val is not None:
                            pred_D_complete.append(val)
                for pair in itertools.zip_longest(nan_lists, pred_H):
                    for val in pair:
                        if val is not None:
                            pred_H_complete.append(val)
                for pair in itertools.zip_longest(nan_lists, pred_Z):
                    for val in pair:
                        if val is not None:
                            pred_Z_complete.append(val)
            else:
                pred_D_complete = []
                pred_H_complete = []
                pred_Z_complete = []
                for pair in itertools.zip_longest(pred_D, nan_lists):
                    for val in pair:
                        if val is not None:
                            pred_D_complete.append(val)
                for pair in itertools.zip_longest(pred_H, nan_lists):
                    for val in pair:
                        if val is not None:
                            pred_H_complete.append(val)
                for pair in itertools.zip_longest(pred_Z, nan_lists):
                    for val in pair:
                        if val is not None:
                            pred_Z_complete.append(val)
                
            # Flatten the lists
            pred_D_complete = [val for sublist in pred_D_complete for val in sublist]
            pred_H_complete = [val for sublist in pred_H_complete for val in sublist]
            pred_Z_complete = [val for sublist in pred_Z_complete for val in sublist]

        else:
             pred_D_complete = pred_D[0]
             pred_H_complete = pred_H[0]
             pred_Z_complete = pred_Z[0]

        #Plot DHZ original y predicciones
        plot_subplots_DHZ(pred_D_complete, pred_H_complete, pred_Z_complete, original_data_copy, min_file, 'Data Original - ', 1)

        D_anomalies, H_anomalies, Z_anomalies = [], [], []
        positions_anomalies = []
        anomaly_threshold = []

        
        # Nueva lista para almacenar los non-empty arrays
        filtered_X = []

        for arr in original_data_filtered_none:
            # Verificando si no es vacio
            if arr.size > 0:
                filtered_X.append(arr)

        
        original_data_filtered_none = filtered_X
        
        # Obtenemos los posibles indices de anomalias
        for i in range(len(original_data_filtered_none)):
            D_anomalies_i, H_anomalies_i, Z_anomalies_i, anomaly_threshold_i = detection_anomalies(pred_D_sub_df[i], pred_H_sub_df[i], pred_Z_sub_df[i], original_data_filtered_none[i], tolerance)
            positions_anomalies_i = [D_anomalies_i, H_anomalies_i, Z_anomalies_i]
            D_anomalies.append(D_anomalies_i)
            H_anomalies.append(H_anomalies_i)
            Z_anomalies.append(Z_anomalies_i)
            positions_anomalies.append(positions_anomalies_i)
            anomaly_threshold.append(anomaly_threshold_i)
            
        
        ev_D, ev_H, ev_Z = [], [], []

        for i in range(len(original_data_filtered_none)):
            evi_D, evi_H, evi_Z = corrected_data(model, original_data_filtered_none[i], sequence_length, anomaly_threshold[i], scaler, tolerance, max_consecutives, positions_anomalies[i])
            ev_D.append(evi_D)
            ev_H.append(evi_H)
            ev_Z.append(evi_Z)
        
        if len(noneData_consecutive_filtered) != 0:
            nan_lists = create_nan_lists(noneData_consecutive_filtered)
                        
            if noneData_consecutive_filtered[0][1] == 0:
                ev_D_complete = []
                ev_H_complete = []
                ev_Z_complete = []
                for pair in itertools.zip_longest(nan_lists, ev_D):
                    for val in pair:
                        if val is not None:
                            ev_D_complete.append(val)
                for pair in itertools.zip_longest(nan_lists, ev_H):
                    for val in pair:
                        if val is not None:
                            ev_H_complete.append(val)
                for pair in itertools.zip_longest(nan_lists, ev_Z):
                    for val in pair:
                        if val is not None:
                            ev_Z_complete.append(val)
            else:
                ev_D_complete = []
                ev_H_complete = []
                ev_Z_complete = []
                for pair in itertools.zip_longest(ev_D, nan_lists):
                    for val in pair:
                        if val is not None:
                            ev_D_complete.append(val)
                for pair in itertools.zip_longest(ev_H, nan_lists):
                    for val in pair:
                        if val is not None:
                            ev_H_complete.append(val)
                for pair in itertools.zip_longest(ev_Z, nan_lists):
                    for val in pair:
                        if val is not None:
                            ev_Z_complete.append(val)

            ev_D_complete = [val for sublist in ev_D_complete for val in sublist]
            ev_H_complete = [val for sublist in ev_H_complete for val in sublist]
            ev_Z_complete = [val for sublist in ev_Z_complete for val in sublist]

        else:
            ev_D_complete = ev_D[0]
            ev_H_complete = ev_H[0]
            ev_Z_complete = ev_Z[0]

        # Ya se tiene los errores puntuales corregidos y se procede a graficar
        # Si se comenta esta línea no habria problema en la ejecucion y seria mas rapida
        # Sin embargo, se mantiene para verificar su correcto funcionamiento
        plot_subplots_DHZ(ev_D_complete, ev_H_complete, ev_Z_complete, original_data_copy2, min_file, 'Data Corregida - ', 1)


        # Con los valores corregidos, se obtienen I y F 
        ev_I_complete = []
        ev_F_complete = []
        for i in range(len(ev_D_complete)):
            ev_I_complete.append(calculate_I(ev_H_complete[i], ev_Z_complete[i]))
            ev_F_complete.append(calculate_F(ev_D_complete[i], ev_H_complete[i], ev_Z_complete[i]))

        # La falta de datos se rellena con 9s en el archivo con data corregida
        ev_D_complete = np.where(np.isnan(ev_D_complete), 99.9999, ev_D_complete)
        ev_H_complete = np.where(np.isnan(ev_H_complete), 99999.9, ev_H_complete)
        ev_Z_complete = np.where(np.isnan(ev_Z_complete), 99999.9, ev_Z_complete)
        ev_I_complete = np.where(np.isnan(ev_I_complete), 99.9999, ev_I_complete)
        ev_F_complete = np.where(np.isnan(ev_F_complete), 99999.9, ev_F_complete)

        save_dataCorr_to_min_file(ev_D_complete, ev_H_complete, ev_Z_complete, ev_I_complete, ev_F_complete, file_path)
                
    return 

def calculate_I(H, Z):
    return np.arctan(Z / H) * ( 180 / np.pi )
def calculate_F(D, H, Z):
    D = D * 180 / np.pi
    X = H * np.cos(D)
    Y = H * np.sin(D)
    return np.sqrt(X**2 + Y**2 + Z**2)


def save_dataCorr_to_min_file(ev_D_complete, ev_H_complete, ev_Z_complete, ev_I_complete, ev_F_complete, min_file_path):
    # Leyendo el archivo .min
    with open(min_file_path, 'r') as f:
        lines = f.readlines()

    # Remplazando 'Reported' por 'Corrected' en la primera linea
    lines[0] = lines[0].replace('Reported', 'Corrected')

    # Actualizando los valores numericos y arreglando el espaciado
    for i in range(5, len(lines)):
        parts = lines[i].split()
        parts[5] = f"{ev_D_complete[i-5]:8.4f}"
        if i == 5:  
            parts[6] = f"{ev_H_complete[i-5]:8.1f}"
            parts[7] = f"{ev_Z_complete[i-5]:7.1f}"
        else:
            parts[6] = f"{ev_H_complete[i-5]:8.1f}"
            parts[7] = f"{ev_Z_complete[i-5]:8.1f}"
        parts[8] = f"{ev_I_complete[i-5]:8.4f}"
        parts[9] = f"{ev_F_complete[i-5]:8.1f}"
        lines[i] = ' ' + ' '.join(parts) + '\n' 
        break

    # Escribiendo las lineas actualizadas
    with open(min_file_path, 'w') as f:
        f.writelines(lines)

    print("Data corregida guardada en el archivo min!")
    print("Correcciones: Anomalias puntuales y falta de datos completada con 9s")



############################################################################################
##### Ejecucion del programa: Archivos min reportados -> Archivos min corregidos ###########
############################################################################################



def main():
    columns = ['D(deg)', 'H(nT)', 'Z(nT)']
        
    # Nombre del archivo del modelo
    model_filename = f'LSTM2_combined_magnitudes_model.h5'

    # Estableciendo parametros y llamando al modelo
    scaler = MinMaxScaler()
    sequence_length = 10 # Valor fijo, determinado al entrenar el modelo

    tolerance = 15  # Este valor indica la tolerancia para la deteccion de anomalias
                    # En porcentaje, si un valor varía más del tolerance% es una anomalía
    model = load_model(model_filename)

    # Ruta para carpetas A y B
    # A : Carpeta donde se ubican los archivos min reportados
    # B : Carpeta donde se almacenaran los archivos corergidos
    folder_a = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TEST'
    folder_b = 'C:\\Users\\Practicas-IGP\\Documents\\LSTM-CORRECTION\\TEST(complete)'
    complete_folder9(folder_a, folder_b)

    # Para realizar el procesamiento de los archivos
    evaluate_folder(folder_b, sequence_length, columns, model, scaler, tolerance)

    
    
if __name__ == "__main__":
    main()
