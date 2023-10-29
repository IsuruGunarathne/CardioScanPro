import streamlit as st
import pandas as pd
from scipy.io import loadmat
import plotly.express as px
import utils
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling1D,
    Dense,
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.keras.models import load_model

# Title
st.title("Cardio Scan Pro")
st.write("Welcome to Cardio Scan Pro. A tool to help you diagnose heart diseases.")
st.write("Please upload the ECG recording and header files below.")


# anomalies df
anomalies_df = pd.read_csv("Dx_map.csv")
details_df = pd.read_csv("Dx_map_with_details.csv")


# Function definitions
def read_hea_file(file):
    header_info = file.readlines()

    return header_info


# Table sketch


# File Uploaders

# collecting mat file from user
mat_file = st.file_uploader(
    "ECG Recording file",
    type=["mat"],
    accept_multiple_files=False,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    disabled=False,
    label_visibility="visible",
)

# collecting hea file from user
hea_file = st.file_uploader(
    "ECG Header file",
    type=["hea"],
    accept_multiple_files=False,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    disabled=False,
    label_visibility="visible",
)

# read the file and convert to dataframe
if mat_file is not None and hea_file is not None:
    data = loadmat(mat_file)
    df = pd.DataFrame(data["val"])
    # st.write(df)

    array = data["val"]
    # the data contains 12 time series, each with 7500 data points
    # plot the first 1000 data points of the first time series

    st.write("First 1000 data points of Lead 1")
    fig = px.line(df.iloc[0, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 2")
    fig = px.line(df.iloc[1, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 3")
    fig = px.line(df.iloc[2, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 4")
    fig = px.line(df.iloc[3, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 5")
    fig = px.line(df.iloc[4, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 6")
    fig = px.line(df.iloc[5, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 7")
    fig = px.line(df.iloc[6, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 8")
    fig = px.line(df.iloc[7, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 9")
    fig = px.line(df.iloc[8, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 10")
    fig = px.line(df.iloc[9, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 11")
    fig = px.line(df.iloc[10, 0:1000])
    st.plotly_chart(fig)

    st.write("First 1000 data points of Lead 12")
    fig = px.line(df.iloc[11, 0:1000])
    st.plotly_chart(fig)

    # read the header file and print data

    # st.write("Header file data")
    header_info = read_hea_file(hea_file)
    freq = int(header_info[0].split()[2])
    age = int(header_info[13].split()[-1])
    sex = str(header_info[14].split()[-1])
    sex = sex[2:-1]
    # st.write(header_info)

    model = load_model("CardioScanPro_model_light_weight.h5")

    processed_array = utils.process_input(array, freq)
    res1 = model.predict(processed_array)
    df_probs = utils.get_best_(list(res1[0]), anomalies_df)

    # Patient details
    output = "The patient is a " + str(age) + " year old " + sex

    # Result heading
    st.header("Results of the ECG scan")
    # Display the table
    st.table(df_probs)

    # Insights
    st.header("Insights")
    st.write(output)
    st.write(
        "The patient is most likely suffering from suffering from "
        + str(df_probs.iloc[0, 0])
        + " ("
        + str(int(df_probs.iloc[0, 1]))
        + "/"
        + str(df_probs.iloc[0, 2])
        + ")"
        + "."
        + " The exact probability is about "
        + str(round((df_probs.iloc[0, 3] * 100), 2))
        + "%"
    )

    # Details
    st.header("Details about the anomaly")
    abbreviated_name = str(df_probs.iloc[0, 2])
    details = details_df[details_df["Abbreviation"] == abbreviated_name]
    st.write("Here are some additional detials about " + str(details.iloc[0, 0]))
    st.write(details.iloc[0, 3])
    st.write("For more detials visit " + str(details.iloc[0, 4]))

st.write("Made with ❤️ by Team 4")
st.write("© 2023 Team 4. All rights reserved.")
st.write("Amila, Isuru, Sulakshi")
