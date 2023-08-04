import os
import librosa
import pickle

TEXT_FILE_NAME = "STFT_GTZAN_dataset.txt"
TEXT_PATH = "/workspace/MusicML/Stage1/audio_file/preprocessed/{}".format(TEXT_FILE_NAME)
DATASET_PATH = "/workspace/MusicML/Stage1/audio_file/raw_imported/GTZAN_3sec"

SAMPLE_RATE = 22050
DURATION = 3 # measured in seconds
FRAME_SIZE = 1024
HOP_SIZE = 512

def save_stft(dataset_path, txt_path, frame_size, hop_size):
    data = {
        "mapping": ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
        "stft": [],
        "labels": []
    
    }
    i = 0
    for (root, dirs, files) in os.walk(dataset_path):
        # ensure we're processing a genre sub-folder level
        if not root == dataset_path:
            # stft processing
            for file in files:
                file_name = file.split(".")[0]
                file_path = os.path.join(root, file)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                stft_data = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size, center=False)
                stft_data = stft_data.T
                
                data["stft"].append(stft_data.tolist())
                data["labels"].append(data["mapping"].index(file_name))
            print(root.split("/")[-1] + " data has been processed!")    
                
                
    with open(txt_path, "wb") as fp:
        pickle.dump(data, fp)
                
            

if __name__ == "__main__":
    save_stft(DATASET_PATH, TEXT_PATH, FRAME_SIZE, HOP_SIZE)