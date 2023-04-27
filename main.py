import streamlit as st
from datetime import date
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM

import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

#sample data
import seaborn as sns
data = sns.load_dataset('iris')

st.title("Blockchain Price Predictor")
st.markdown("---")
tomorrow_prediction = 234562
st.markdown(f"""### Tomorrow's Prediction: <span style="color:cyan;">{tomorrow_prediction}</span>""", unsafe_allow_html=True)


st.markdown("###")
st.header("LSTM")
sample_fig1 = px.bar(data_frame=data, x="species", y="petal_length", color="species")
st.plotly_chart(sample_fig1, use_container_width=True)

accuracy_lstm = 0.92
st.markdown(f"""### Accuracy: <span style="color:cyan;">{accuracy_lstm}</span>""", unsafe_allow_html=True)


st.markdown("###")
st.header("ARIMA")
sample_fig1 = px.bar(data_frame=data, x="species", y="sepal_width", color="species")
st.plotly_chart(sample_fig1, use_container_width=True)

accuracy_arima = 0.81

st.markdown(f"""### Accuracy: <span style="color:cyan;">{accuracy_arima}</span>""", unsafe_allow_html=True)