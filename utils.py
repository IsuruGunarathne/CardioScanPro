
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib as pl
import numpy as np
import random


#=====================================================================================================

def read_heafile(file_name):
    # Function to read a .hea file and return its content as a list of strings
    # Open the .hea file
    with open(file_name, 'r') as file:
        # Read the content of the .hea file
        hea_content = file.readlines()

    return hea_content

#====================================================================================================

def create_array(hea_content):
    ID = hea_content[0].strip().split()[0]
    
    # Extract 'Age' from .hea file content
    age_info = hea_content[13].strip().split()
    age = int(age_info[2]) if len(age_info) > 2 and age_info[2].isdigit() else 0
    
    # Extract 'Gender' from .hea file content
    gender = hea_content[14].strip().split()[2] if len(hea_content) > 14 else 'Unknown'
    
    # Extract 'Abnormality' from .hea file content
    abnormality = hea_content[15].strip().split()[2] if len(hea_content) > 15 else 'Unknown'
    
    return [ID, age, gender, abnormality]

#===================================================================================================
def create_dataframes(training_directory):
    dataframes = {}

    subdirectories = [subdir for subdir in pl.Path(training_directory).iterdir() if subdir.is_dir()]
    
    for source_folder_path in subdirectories:
        source_folder_name = source_folder_path.name
        columns = ['ID', 'Age', 'Gender', 'Abnormality']
        source_dataframe = pd.DataFrame(columns=columns)
        patient_data = {}  # To collect patient information
        
        for subdir in source_folder_path.iterdir():
            if subdir.is_dir():
                data_dir = pl.Path(subdir)
                header_files = list(data_dir.glob('*.hea'))

                for header_file in header_files:
                    header_path = data_dir.joinpath(header_file.name)
                    hea_content = read_heafile(header_path)
                    patient_info = create_array(hea_content)
                    patient_id = patient_info[0]
                    
                    # Collect patient information
                    for i, column_name in enumerate(['Age', 'Gender', 'Abnormality']):
                        patient_data.setdefault(patient_id, {})[column_name] = patient_info[i + 1]
                        
        # Create a list of patient data dictionaries
        patient_rows = []
        for patient_id, info in patient_data.items():
            row = {'ID': patient_id, 'Age': info.get('Age'), 'Gender': info.get('Gender'), 'Abnormality': info.get('Abnormality')}
            patient_rows.append(row)
            
        # Concatenate patient data into the dataframe
        source_dataframe = pd.concat([source_dataframe, pd.DataFrame(patient_rows)])
        
        dataframes[f'{source_folder_name}_df'] = source_dataframe
        
    return dataframes


#===================================================================================================

# Function for normializing the wave 
#parameters 
#  wave form representing the array
#  frequency for normalization
#  frequency of the waveform
def normalize_wave(array,nrm_freq,freq):
    factor = round(freq/nrm_freq)
    normalized_array = []
    for ele in array:
        new_ele = ele[::factor]
        normalized_array.append(new_ele)
    return len(normalized_array[0]),np.array(normalized_array)


#===================================================================================================

def normalize_mats(dir_path):
    # This function will iterate thorugh a data directory and return a list of 
    # nomlized waveforms for the ECG's in that directory
    normalized_waves = []
    lengths = []
    # Iterating through the subdirectories inside the given directory
    for subdir in pl.Path(dir_path).iterdir():
        if subdir.is_dir():
            
            data_dir = pl.Path(subdir)
            head_file_list = list(data_dir.glob('*.hea'))
            mat_file_list = list(data_dir.glob('*.mat'))
            for i in range(len(head_file_list)):
                head_file_path = data_dir.joinpath(head_file_list[i].name)
                mat_file_path = data_dir.joinpath(mat_file_list[i].name)

                data = scipy.io.loadmat(mat_file_path)['val']
                current_frequency = int(read_heafile(head_file_path)[0].split()[2])
                length,nomralized_wave = normalize_wave(data,250,current_frequency)
                normalized_waves.append(nomralized_wave)
                lengths.append(length)
    return lengths,normalized_waves


#===================================================================================================

def read_heads(dir_path):
    # this function will create a arrays of frequencies,number of points and combination of them
    freq_array = []
    pts_array = []
    both = []
    # Iterating through the subdirectories inside the given directory
    for subdir in pl.Path(dir_path).iterdir():
        if subdir.is_dir():
            
            data_dir = pl.Path(subdir)
            file_list = list(data_dir.glob('*.hea'))
    
            for file in file_list:
                file_path = data_dir.joinpath(file.name)
                data = read_heafile(file_path)
                freq = int(data[0].split()[2])
                points = int(data[0].split()[3].strip('\n'))
                freq_array.append(freq)
                pts_array.append(points)
                both.append([freq,points])
        
    return freq_array,pts_array,both


#===================================================================================================

def create_anomalies_array(data):
    """
    This function will take a .csv file as the input.
    It will create a array containing all the anomalies
    """
    anomalies_array = []
    
    for index,row in data.iterrows():
        anomalies_array.append(row['SNOMED CT Code'])
    
    return anomalies_array

#===================================================================================================

def create_single_output_array(array,anomalies):
    """
    This will take the anomalies array and the array of anomalies of a patient
    This will output an array conatinimg binary values.
    It represents the 1 when a patient has the relavent anomaly , otherwise 0
    """
    data = create_anomalies_array(anomalies)
    
    for i in range(len(data)):
        if(data[i] in array):
            data[i] = 1
        else:
            data[i] = 0
    return data

#===================================================================================================

def create_output_array(df,anomalies):
    """
    This will take anomalies array and a data frame as the input
    This will output the Y data set 
    """
    Y = []
    
    for index,row in df.iterrows():
        # Create the anomalies array for the relavent row
        # --------code here---------
        array = []
        
        output = create_single_output_array(array,anomalies)
        Y.append(output)
        
    return np.array(Y)


#===================================================================================================

def create_y_array(df,data,source_file):
    """
    This function will take a dataframe(heads),csv of anomalies and a list of source files
    This will output the Y array for the given source files(Y array is the array conatining training labels)
    """
    Y = []
    for ele in source_file:
        y = create_output_array(df[ele],data)
        Y = Y + y
    return Y

#===================================================================================================

def create_x_array(source_file):
    """
    This function will take a list of source files
    This will output the X array for the given source files(X array is the array conatining training X data)
    """

    X = []
    lengths = []
    for ele in source_file:
        length,array = normalize_mats('training/' + ele)
        lengths = lengths + length
        X = X + array
    return X,lengths


#===================================================================================================

def equalizing_wave_array(x_copy):
    """
    This function will take the X array and equalize the length of the waves
    """
    x_copy_new = []
    for ele in x_copy:
    
        size = len(ele[0])

        # If the size of the teh array is less than 2617 it will add noice at the end and begining 
        if(size < 2617):
        
            start = round((2617 - size)/2)
            end = 2617 - size - start
        
            new_array = []
        
            for data in ele:

                lower_bound,upper_bound = min(data),max(data)
            
                start_list = [random.randint(lower_bound, upper_bound) for _ in range(start)]
                end_list = [random.randint(lower_bound, upper_bound) for _ in range(end)]
            
                new_sub_array = np.array(start_list + list(data) + end_list)
                new_array.append(new_sub_array)
   
            x_copy_new.append(new_array)
        
        # Else it will simmply catoff the extra part from the begining and the end
        else:
            extra = size - 2617
            half_extra = round(extra)
        
            new_array = []
        
            for data in ele:
                new_sub_array = list(data)[(half_extra-1):(half_extra + 2616)]
                new_array.append(new_sub_array)
            x_copy_new.append(new_array)


    return x_copy_new



#===================================================================================================

def process_input(array,freq):

    """
    This function will process the input so that it could be fed in to the model and do the prediction
    When the input array and the frequency is given it will return a array of size (1,12,2617) by
    Normalizing and Reshaping the wave
    """
    
    size,normlaized_wave = normalize_wave(array,250,freq)
    
    
    if(size < 2617):
        
        start = round((2617 - size)/2)
        end = 2617 - size - start
        
        new_array = []
        for data in normlaized_wave:
            
            lower_bound,upper_bound = min(data),max(data)
            
            start_list = [random.randint(lower_bound, upper_bound) for _ in range(start)]
            end_list = [random.randint(lower_bound, upper_bound) for _ in range(end)]
            
            new_sub_array = np.array(start_list + list(data) + end_list)
            new_array.append(new_sub_array)
   
        return np.expand_dims(np.array(new_array),axis = 0)
    else:
        
        extra = size - 2617
        half_extra = round(extra/2)
        
        new_array = []
        
        for data in normlaized_wave:
            new_sub_array = list(data)[(half_extra-1):(half_extra + 2616)]
            new_array.append(new_sub_array)
        
        return np.expand_dims(np.array(new_array),axis = 0)
    

#===================================================================================================

def get_best_(array,df):
    table_data = {
        'Abnormality' : [],
        'SNOMED CT Code' : [],
        'Abbrevation' : [],
        'Probability' : []
    }

    sorted_array = sorted(array)[::-1]
    for ele in sorted_array[0:9]:
        index = array.index(ele)
        row_data = df.iloc[index]
        table_data['Abnormality'].append(row_data['Dx'])
        table_data['SNOMED CT Code'].append(row_data['SNOMED CT Code'])
        table_data['Abbrevation'].append(row_data['Abbreviation'])
        table_data['Probability'].append(ele)

    df = pd.DataFrame(table_data)

    return df