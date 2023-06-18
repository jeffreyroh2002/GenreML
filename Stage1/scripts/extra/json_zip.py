import gzip
import shutil

def compress_json_file(json_file_path, compressed_file_path):
    with open(json_file_path, 'rb') as file:
        with gzip.open(compressed_file_path, 'wb') as compressed_file:
            shutil.copyfileobj(file, compressed_file)

    print(f'Successfully compressed {json_file_path} to {compressed_file_path}.')

# Example usage
json_file_path = '../../audio_file/preprocessed/irmas_noVoice_3sec.json'
compressed_file_path = 'irmas_noVoice_3sec_json.zip'
compress_json_file(json_file_path, compressed_file_path)