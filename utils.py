
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pathlib as pl
import numpy as np

#=====================================================================================================

def read_heafile(file_name):
    # Function to read a .hea file and return its content as a list of strings
    # Open the .hea file
    with open(file_name, 'r') as file:
        # Read the content of the .hea file
        hea_content = file.readlines()

    return hea_content

#====================================================================================================

def create_array(head_file):
    # this function will create an array of the data in the .hea file

    content = read_heafile(head_file)
    ID = content[0].strip('\n').split()[0]
    age = int(content[13].strip('\n').split()[2])
    gender = content[14].strip('\n').split()[2]
    abormalities = content[15].strip('\n').split()[2]
    
    return [ID,age,gender,abormalities]

#Sulakshi- create
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
def create_df(dir_path):
    # this function will build a data frame using the .hea files in a given directory
    
    # Initializing the data frame
    df = pd.DataFrame(columns = ['ID','Age','Gender','Abnormality'])
    
    # Iterating through the subdirectories inside the given directory
    for subdir in pl.Path(dir_path).iterdir():
        if subdir.is_dir():
            
            data_dir = pl.Path(subdir)
            file_list = list(data_dir.glob('*.hea'))
    
            for file in file_list:
                file_path = data_dir.joinpath(file.name)
                print(file_path)
                data = create_array(file_path)
                df.loc[len(df)] = data
        
    return df


#===================================================================================================
#sulakshi_datframes

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

# Call the function with your specific directory
directory_path = r'C:\Users\ASUS\Desktop\CardioData\CardioData\training'
result_dataframes = create_dataframes(directory_path)
print(result_dataframes)

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

# Call the function with your specific directory
directory_path = r'C:\Users\ASUS\Desktop\CardioData\CardioData\training'
result_dataframes = create_dataframes(directory_path)
print(result_dataframes)