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
#print(df.head())


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

df = df.drop(labels='filename', axis=1)
#print(df.head())

#converting text data to numerical data. es) blues = 0, country = 1, etc

class_list = df.iloc[:, -1]
#print(class_list)
convertor = LabelEncoder()    # maybe use one-hot encoder instead?
y = convertor.fit_transform(class_list)   #labels (expectation) represented in integers

#Scaling the Feature --> need to research more on this
from sklearn.preprocessing import StandardScaler
fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype = float))  # 
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



