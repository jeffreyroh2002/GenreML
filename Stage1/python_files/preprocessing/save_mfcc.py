import json
import os
import math
import librosa

DATASET_PATH = "genre_dataset_reduced"
JSON_PATH = "data.json"

def save_mfcc(data_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
	
	#data dictionary
	data = {
		"mapping": [],
		"mfcc":[],
		"labels": []
	}
	
	#loop through all genres
	for i, (dirpath, dirnames, filenames) in enumerate (os.walk(dataset_path)):
		
		# ensure we're processing a genre sub-folder level
		if dirpath is not dataset_path: 
			
			
		
	
	