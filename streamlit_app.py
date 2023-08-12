import streamlit as st
import pandas as pd
import numpy as np, os, sys
from scipy.io import loadmat
import plotly.express as px

# Title
st.title("Cardio Scan Pro")
st.write("This is Cardio Scan Pro")


# Function definitions
def read_hea_file(file):
    header_info = file.readlines()

    return header_info


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
if mat_file is not None:
    data = loadmat(mat_file)
    df = pd.DataFrame(data["val"])
    st.write(df)

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


# read the header file and convert to dataframe
if hea_file is not None:
    header_info = read_hea_file(hea_file)
    st.write(header_info)
