
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