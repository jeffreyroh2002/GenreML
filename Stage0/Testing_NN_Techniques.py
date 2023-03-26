import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import os
import pickle
import librosa
import librosa.display
import IPython
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('../input_data/features_3_sec.csv')
print(df.head())


#data , sr = librosa.load(audio_recording= "../input_data/features_3_sec.csv/country/country

"""
IPython.display.Audio(data, rate=sr)

#Short-time Fourier transform
stft = librosa.stft(data)
stft_db = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(14,6))
librosa.display.specshow(stft, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

"""



