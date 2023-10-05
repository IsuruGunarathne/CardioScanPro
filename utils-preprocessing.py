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



