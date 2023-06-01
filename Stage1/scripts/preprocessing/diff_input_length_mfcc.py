import json
import os
import math
import librosa

DATASET_PATH = "../../audio_file/raw_imported/mood_soundtracks/mp3_files"
JSON_FILE_NAME = "mood_soundtracks0518.json"
JSON_PATH = "../../audio_file/preprocessed/{}".format(JSON_FILE_NAME)

SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
NUM_SEGMENTS = 5

def save_mfcc(dataset_path, json_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, num_segments=NUM_SEGMENTS):

    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    # Loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            # Save semantic label
            dirpath_components = dirpath.split("/")
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # Process files for specific genre
            for f in filenames:

                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # Calculate the number of samples per segment dynamically
                num_samples_per_segment = math.ceil(len(signal) / num_segments)

                # Process segments, extract MFCC, and store data
                for s in range(num_segments):
                    start_sample = s * num_samples_per_segment
                    finish_sample = min((s + 1) * num_samples_per_segment, len(signal))

                    mfcc = librosa.feature.mfcc(
                        y=signal[start_sample:finish_sample],
                        sr=sr,
                        n_fft=n_fft,
                        n_mfcc=n_mfcc,
                        hop_length=hop_length
                    )
                    mfcc = mfcc.T

                    # Store MFCC for segment if it has the expected length
                    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)
                    if len(mfcc) >= expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, s))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
