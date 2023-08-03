from pydub import AudioSegment
import os
import random

PATH = "/workspace/MusicML2/Stage1/audio_file/raw_imported/genres_original"
OUT_PATH = "/workspace/MusicML2/Stage1/audio_file/raw_imported/GTZAN_3sec"


# /workspace/MusicML2/Stage1/audio_file/raw_imported/Data/genres_original/blues/blues.00000.wav
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

def split_wav_file(audio_path):
    random_seed = []
    f_path = audio_path.split("/")
    
    genre = f_path[-2]
    genre_path = os.path.join(OUT_PATH,genre)
    if not os.path.exists(genre_path):
        os.makedirs(genre_path)
    
    for i in range(3):
        random_seed.append(random.randint(0, 18))
    
    for i in range(19):
        if i in random_seed: 
            t1 = i * 1.5
            t2 = (i+2) * 1.5
            t1 = t1 * 1000 #Works in milliseconds
            t2 = t2 * 1000


            file_name = f_path[-1]
            file_name = file_name.split(".")
            name = file_name[1] + "-" + str(i+1).zfill(2)
            file_name[1] = name
            file_name = ".".join(file_name)
            newAudio = AudioSegment.from_wav(audio_path)
            newAudio = newAudio[t1:t2]
            newAudio.export(os.path.join(genre_path,file_name), format="wav") #Exports to a wav file in the current path.
    
    # print(genre + " has been splitted!");
    
if __name__ == "__main__":
    for (root, dirs, files) in os.walk(PATH):
        for filename in files:
            split_wav_file(root + "/" + filename)
