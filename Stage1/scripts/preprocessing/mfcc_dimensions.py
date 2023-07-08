import json
import os
import math
import librosa


DATASET_PATH = "../../audio_file/raw_imported/genres_original/blues/blues.00000.wav"
JSON_FILE_NAME = "mfcc_testing.json"
JSON_PATH = "../../audio_file/preprocessed/{}".format(JSON_FILE_NAME)

SAMPLE_RATE = 22050
DURATION = 30 # measured in seconds for GTZAN Dataset
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

"""
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

# show plots
plt.show()
"""
json_path = JSON_PATH 
n_mfcc=13
n_fft=2048
hop_length=512
num_segments=5

num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)

#need number of vectors for mfcc extraction to be equal for each segment
expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
    
signal, sr = librosa.load(DATASET_PATH, sr=SAMPLE_RATE)

#process segments extracting mfcc and storing data
for s in range(num_segments):
	start_sample = num_samples_per_segment * s
	finish_sample = start_sample + num_samples_per_segment

	mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample],
													   sr = sr,
													   n_fft=n_fft,
													   n_mfcc = n_mfcc,
													   hop_length=hop_length)
	mfcc = mfcc.T
	print(mfcc.shape)
