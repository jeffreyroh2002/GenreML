import os
import sys
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

PATH = "/workspace/MusicML3/Stage1/audio_file/preprocessed/corrupted_file_idx.npz"
DIR_NAME = "/workspace/MusicML3/Stage1/audio_file/raw_imported/GTZAN_30SongsEach/genres_original"

def pre_setting():
    # Save audio paths and labels
    audio_paths = []
    # audio_dict = {}
    audio_label = []
    # Print all the files in different directories
    for root, dirs, files in os.walk(DIR_NAME, topdown=False):
        for filenames in files:
            if filenames.find('.wav') != -1:

                audio_paths.append(os.path.join(root, filenames))
                filenames = filenames.split('.', 1)
                filenames = filenames[0]
                audio_label.append(filenames)
    audio_paths = np.array(audio_paths)
    audio_label = np.array(audio_label)
    print(audio_paths.shape)
    print("1")

    return audio_paths, audio_label


def check_audio_file(audio_paths):
    # Create empty arrays to save the features
    AllSpec = np.empty([310, 1025, 1293])
    AllMel = np.empty([310, 128, 1293])
    AllMfcc = np.empty([310, 10, 1293])
    AllZcr = np.empty([310, 1293])
    AllCen = np.empty([310, 1293])
    AllChroma = np.empty([310, 12, 1293])

    print("2")
    count = 0
    bad_index = []
    for i in tqdm(range(len(audio_paths))):
        try:
            p = audio_paths[i]
            y, sr = librosa.load(p)
            # For Spectrogram
            X = librosa.stft(y)
            Xdb = librosa.amplitude_to_db(abs(X))
            AllSpec[i] = Xdb

            # Mel-Spectrogram
            M = librosa.feature.melspectrogram(y=y)
            M_db = librosa.power_to_db(M)
            AllMel[i] = M_db

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
            AllMfcc[i] = mfcc

            # Zero-crossing rate
            # zcr = librosa.feature.zero_crossing_rate(y)[0]
            # AllZcr[i] = zcr

            # # Spectral centroid
            # sp_cen = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            # AllCen[i] = sp_cen

            # # Chromagram
            # chroma_stft = librosa.feature.chroma_stft(
            #     y=y, sr=sr, n_chroma=12, n_fft=4096)
            # AllChroma[i] = chroma_stft

        except Exception as e:
            bad_index.append(i)

    return bad_index


if __name__ == "__main__":
    audio_paths, audio_label = pre_setting()
    bad_index = check_audio_file(audio_paths)
    np.savez_compressed(PATH, bad_index=bad_index,
                        audio_paths=audio_paths, audio_label=audio_label)
    print("Successfully saved data")
