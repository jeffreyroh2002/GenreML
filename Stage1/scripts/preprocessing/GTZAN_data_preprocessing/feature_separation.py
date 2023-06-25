import os
import sys
import cv2
import seaborn as sn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = "/workspace/MusicML/Stage1/audio_file/preprocessed/GTZAN_features"

# load the .npz file of features
f = np.load(PATH + "/MusicFeatures.npz")
S = f['spec']
mfcc = f['mfcc']
mel = f['mel']
chroma = f['chroma']
y = f['target']
print(S.shape)
print(mfcc.shape)
print(mel.shape)
print(chroma.shape)
print(y.shape)

# split train_test data
S_train, S_test, mfcc_train, mfcc_test, mel_train, mel_test, chroma_train, chroma_test, y_train, y_test = train_test_split(
    S, mfcc, mel, chroma, y, test_size=0.2)

# Resizing and Reshaping Data
# MFCC
newtrain_mfcc = np.empty((mfcc_train.shape[0], 120, 600))
newtest_mfcc = np.empty((mfcc_test.shape[0], 120, 600))

for i in range(mfcc_train.shape[0]):

    curr = mfcc_train[i]
    curr = cv2.resize(curr, (600, 120))
    newtrain_mfcc[i] = curr

mfcc_train = newtrain_mfcc

for i in range(mfcc_test.shape[0]):

    curr = mfcc_test[i]
    curr = cv2.resize(curr, (600, 120))
    newtest_mfcc[i] = curr

mfcc_test = newtest_mfcc


maximum = np.amax(mfcc_train)
# mfcc_train = mfcc_train/maximum
# mfcc_test = mfcc_test/maximum

mfcc_train = mfcc_train.astype(np.float32)
mfcc_test = mfcc_test.astype(np.float32)


N, row, col = mfcc_train.shape
mfcc_train = mfcc_train.reshape((N, row, col, 1))

N, row, col = mfcc_test.shape
mfcc_test = mfcc_test.reshape((N, row, col, 1))
# print(mfcc_train.shape, mfcc_test.shape)


mean_data = np.mean(mfcc_train)
std_data = np.std(mfcc_train)

mfcc_train = (mfcc_train - mean_data) / std_data
mfcc_test = (mfcc_test - mean_data) / std_data

# print(np.amin(mfcc_train), np.amax(mfcc_train))
# print(np.amin(mfcc_test), np.amax(mfcc_test))

# Spectrogram
maximum1 = np.amax(S_train)
S_train = S_train/np.amax(maximum1)
S_test = S_test/np.amax(maximum1)

S_train = S_train.astype(np.float32)
S_test = S_test.astype(np.float32)

N, row, col = S_train.shape
S_train = S_train.reshape((N, row, col, 1))

N, row, col = S_test.shape
S_test = S_test.reshape((N, row, col, 1))
# print(S_train.shape, S_test.shape)


# Mel-Spectrogram

maximum = np.amax(mel_train)
mel_train = mel_train/np.amax(maximum)
mel_test = mel_test/np.amax(maximum)

mel_train = mel_train.astype(np.float32)
mel_test = mel_test.astype(np.float32)

N, row, col = mel_train.shape
mel_train = mel_train.reshape((N, row, col, 1))

N, row, col = mel_test.shape
mel_test = mel_test.reshape((N, row, col, 1))
# print(mel_train.shape, mel_test.shape)

print("Shape of data")
print(mfcc_train.shape, mfcc_test.shape)
print(S_train.shape, S_test.shape)
print(mel_train.shape, mel_test.shape)

# Save training and testing features in npz file
# Save Spectrogram train, test
np.savez_compressed(PATH + "/new_spectrogram_train_test.npz",
                    S_train=S_train, S_test=S_test, y_train=y_train, y_test=y_test)

# Save mfcc train, test
print(mfcc_train.shape)
print(mfcc_test.shape)
np.savez_compressed(PATH + "/new_mfcc_train_test.npz", mfcc_train=mfcc_train,
                    mfcc_test=mfcc_test, y_train=y_train, y_test=y_test)

# Save Mel-Spectrogram train, test
np.savez_compressed(PATH + "/new_mel_train_test.npz", mel_train=mel_train,
                    mel_test=mel_test, y_train=y_train, y_test=y_test)
