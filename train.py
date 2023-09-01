import utils
import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
import model as mdl

data = pd.read_csv('Dx_map.csv')

df = utils.create_dataframes('training')


srce_files_df = ['cpsc_2018_df', 'cpsc_2018_extra_df', 'georgia_df', 'ptb_df', 'ptb-xl_df', 'st_petersburg_incart_df']


srce_files = ['cpsc_2018', 'cpsc_2018_extra', 'georgia', 'ptb', 'ptb-xl', 'st_petersburg_incart']

#================================================================================================================
X,lengths = utils.create_y_array(srce_files)
Y = utils.create_y_array(df,data,srce_files_df)


#================================================================================================================
# Removing the outliers / Unwanted data
new_sizes = []
for i in range(len(lengths)):
    if(lengths[i] < 1000 or lengths[i] > 5000):
        Y[i] = 0
        X[i] = 0
    else:
        new_sizes.append(lengths[i])

# Modifying the arrays after removing unwanted values
X = [item for item in X if type(item) != int]
Y = [item for item in Y if type(item) != int]


# Adding noice to the  data to make it 2617 points long
X = utils.equalizing_wave_array(X)

# Convering the list of arrays to numpy arrays
for i in range(len(X)):
    X[i] = np.array(X[i])

for i in range(len(Y)):
    Y[i] = np.array(Y[i])

# Splitting the data into train and test    
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size=0.1, random_state=42)


# Getting the input shape and number of classes(output)
input_shape = (X_train.shape[1], X_train.shape[2])  # Shape: (sequence_length, num_leads)
num_classes = y_train.shape[1]  # Number of anomaly classes


# Creating the model
resnet_model = mdl.ResNet_model(input_shape,num_classes)

# Training the model
trained_model,accuracy_results_loss_results = mdl.model_train(X_train,y_train,resnet_model,5,15)