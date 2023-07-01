# Import libraries
import os
import sys
import numpy as np
import pandas as pd
import librosa
import IPython.display as ipd
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import json
import tensorflow

CORRUPT_IDX_PATH = "/workspace/MusicML/Stage1/audio_file/preprocessed/corrupted_file_idx.npz"
PATH = "/workspace/MusicML/Stage1/audio_file/preprocessed/GTZAN_features"

os.makedirs(PATH)

def feature_extraction(bad_index, audio_paths, audio_label):
    file_num = len(audio_paths)
    # Create empty arrays to save the features
    AllSpec = np.empty([file_num, 1025, 1293])
    AllMel = np.empty([file_num, 128, 1293])
    AllMfcc = np.empty([file_num, 10, 1293])
    AllZcr = np.empty([file_num, 1293])
    AllCen = np.empty([file_num, 1293])
    AllChroma = np.empty([file_num, 12, 1293])

    print("2")
    count = 0
    for i in tqdm(range(len(audio_paths))):
        if i > AllSpec.shape[0] - 1:
            new_i = file_num - i
            new_i = AllSpec.shape[0] - new_i
        else:
            new_i = i
        if i in bad_index:
            AllSpec = np.delete(AllSpec, new_i, 0)
            AllMel = np.delete(AllMel, new_i, 0)
            AllMfcc = np.delete(AllMfcc, new_i, 0)
            AllZcr = np.delete(AllZcr, new_i, 0)
            AllCen = np.delete(AllCen, new_i, 0)
            AllChroma = np.delete(AllChroma, new_i, 0)
            continue
        path = audio_paths[i]
        y, sr = librosa.load(path)
        # For Spectrogram
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        AllSpec[new_i] = Xdb

        # Mel-Spectrogram
        M = librosa.feature.melspectrogram(y=y)
        M_db = librosa.power_to_db(M)
        AllMel[new_i] = M_db

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
        AllMfcc[new_i] = mfcc

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        AllZcr[new_i] = zcr

        # Spectral centroid
        sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        AllCen[new_i] = sp_cen

        # Chromagram
        chroma_stft = librosa.feature.chroma_stft(
            y=y, sr=sr, n_chroma=12, n_fft=4096)
        AllChroma[new_i] = chroma_stft

    print("4")
    # Convert to float32
    AllSpec = AllSpec.astype(np.float32)
    AllMel = AllMel.astype(np.float32)
    AllMfcc = AllMfcc.astype(np.float32)
    AllZcr = AllZcr.astype(np.float32)
    AllCen = AllCen.astype(np.float32)
    AllChroma = AllChroma.astype(np.float32)
    print("5")
    # Delete labels at corrupt indices
    audio_label = np.delete(audio_label, bad_index)
    print("6")
    # Convert labels from string to numerical
    audio_label[audio_label == 'blues'] = 0
    audio_label[audio_label == 'classical'] = 1
    audio_label[audio_label == 'country'] = 2
    audio_label[audio_label == 'disco'] = 3
    audio_label[audio_label == 'hiphop'] = 4
    audio_label[audio_label == 'jazz'] = 5
    audio_label[audio_label == 'metal'] = 6
    audio_label[audio_label == 'pop'] = 7
    audio_label[audio_label == 'reggae'] = 8
    audio_label[audio_label == 'rock'] = 9
    audio_label = [int(i) for i in audio_label]
    audio_label = np.array(audio_label)
    print("7")
    # Convert labels from numerical to categorical data
    y = tensorflow.keras.utils.to_categorical(
        audio_label, num_classes=10, dtype="int32")
    print("8")

    # Save all the features and labels in a .npz file
    np.savez_compressed(PATH + "/MusicFeatures.npz", spec=AllSpec, mel=AllMel,
                        mfcc=AllMfcc, zcr=AllZcr, cen=AllCen, chroma=AllChroma, target=y)
    print("Successfully Saved File at: ", PATH + "/MusicFeatures.npz")


if __name__ == "__main__":
    f = np.load(CORRUPT_IDX_PATH)
    bad_index = f["bad_index"]
    audio_paths = f["audio_paths"]
    audio_label = f["audio_label"]
    print(bad_index)

    feature_extraction(bad_index, audio_paths, audio_label)
