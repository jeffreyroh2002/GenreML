import os
import subprocess

# Input and output directories
input_dir = '/path/to/mp3/files/'
output_dir = '/path/to/wav/files/'

# Recursively traverse the directory tree
for root, dirs, files in os.walk(input_dir):
    for file_name in files:
        if file_name.endswith('.mp3'):
            # Input and output file paths
            input_path = os.path.join(root, file_name)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path[:-4] + '.wav')

            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Convert MP3 to WAV using FFmpeg
            command = ['ffmpeg', '-i', input_path, output_path]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            print(f"Converted: {input_path}")

print("Conversion completed.")
