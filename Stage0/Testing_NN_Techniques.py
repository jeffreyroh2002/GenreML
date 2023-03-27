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
from keras.models import Sequential

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

#Building the Model
def trainModel(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer = optimizer,
                  loss = 'sparse_categorical_crossentropy',
                  metrics = 'accuracy'
    )
    return model.fit(X_train, y_train, 
                     validation_data=(X_test, y_test),
                     epochs = epochs, batch_size = batch_size)

def plotValidate(history):
    print("Validation Accuracy", max(history.history["val_accuarcy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()
    
cnn_model = keras.models.Sequential([
    keras.layers.Dense(512, activation = 'relu', input_shape= (X_train.shape[1],)),  #??
    keras.layers.Dropout(0.2),  # heart of CNN

    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(0.2), 
    
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dropout(0.2), 
    
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dropout(0.2), 
    
    keras.layers.Dense(10, activation = 'softmax'),
])
print(cnn_model.summary())
model_history = trainModel(model=cnn_model, epochs=600, optimizer='adam')

#model evaluation
test_loss, test_acc = cnn_model.evaluate(X_test, y_test, batch_size=120)
print("The test Loss is :", test_loss)
print("The Best test Accuracy is :", test_acc*100)
