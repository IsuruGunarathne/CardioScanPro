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

def preprocess_dataframes(result_dataframes):
    cpsc_2018_df = result_dataframes['cpsc_2018_df']
    cpsc_2018_extra_df = result_dataframes['cpsc_2018_extra_df']
    georgia_df = result_dataframes['georgia_df']
    ptb_df = result_dataframes['ptb_df']
    ptb_xl_df = result_dataframes['ptb-xl_df']
    st_petersburg_incart_df = result_dataframes['st_petersburg_incart_df']

    dataframes = [cpsc_2018_df, cpsc_2018_extra_df, georgia_df, ptb_df, ptb_xl_df, st_petersburg_incart_df]

    # Replace invalid age values with NaN (preidentified that outliers are only in ptb_xl_df dataframe)
    invalid_age_indices = ptb_xl_df['Age'] == 300
    ptb_xl_df.loc[invalid_age_indices, 'Age'] = np.nan

    # Replace placeholders with NaN
    for df in dataframes:
        df['Age'] = df['Age'].replace(0, np.nan)
        df['Gender'] = df['Gender'].replace('Unknown', np.nan)
        df['Abnormality'] = df['Abnormality'].replace('Unknown', np.nan)

    return dataframes

# Assuming you have already created the 'result_dataframes' dictionary
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

def fill_missing_ages_with_regression(df):
    # Create a copy of the dataframe to modify
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df_copy = df.copy()
    
    # Create a one-hot encoded matrix for 'Abnormality'
    encoder = OneHotEncoder()
    abnormality_encoded = encoder.fit_transform(df_copy['Abnormality'].values.reshape(-1, 1)).toarray()
    
    # Define features and target variables
    X = abnormality_encoded
    y = df_copy['Age']
    
    # Identify rows with missing 'Age' values
    missing_age_indices = np.isnan(y)
    
    if np.sum(missing_age_indices) == 0:
        return df  # Skip if there are no missing 'Age' values
    
    # Build a linear regression model
    regressor = LinearRegression()
    regressor.fit(X[~missing_age_indices], y[~missing_age_indices])
    
    # Fill missing 'Age' values using the regression model
    for index, row in df.iterrows():
        if np.isnan(row['Age']):
            abnormality = row['Abnormality']
            abnormality_encoded = encoder.transform(np.array(abnormality).reshape(1, -1)).toarray()
            predicted_age = regressor.predict(abnormality_encoded)[0]
            df.loc[index, 'Age'] = predicted_age
    
    return df

# Assuming 'result_dataframes' is a dictionary of DataFrames
# You can call this function for each DataFrame in the dictionary










