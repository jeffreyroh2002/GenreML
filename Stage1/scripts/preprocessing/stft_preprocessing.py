import os
import json
import librosa

JSON_FILE_NAME = "STFT_GTZAN_dataset.json"
JSON_PATH = "/workspace/MusicML2/Stage1/audio_file/preprocessed/{}".format(JSON_FILE_NAME)
DATASET_PATH = "/workspace/MusicML2/Stage1/audio_file/raw_imported/GTZAN_3sec"

SAMPLE_RATE = 22050
DURATION = 3 # measured in seconds
FRAME_SIZE = 1024
HOP_SIZE = 512

def save_stft(dataset_path, json_path, frame_size, hop_size):
    data = {
        "mapping": ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
        "stft": [],
        "labels": []
    
    }
    
    for (root, dirs, files) in os.walk(dataset_path):
        # ensure we're processing a genre sub-folder level
        if not root == dataset_path:
            # stft processing
            for file in files:
                file_name = file.split(".")[0]
                file_path = os.path.join(root, file)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                stft_data = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size, center=False)
                
                data["stft"].append(stft_data.tolist())
                data["labels"].append(data["mapping"].index(file_name))
            print(root.split("/")[-1] + " data has been processed!")    
                
                
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
                
            

if __name__ == "__main__":
    save_stft(DATASET_PATH, JSON_PATH, FRAME_SIZE, HOP_SIZE)