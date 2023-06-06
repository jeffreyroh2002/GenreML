import tarfile

def unzip_tgz_file(file_path, output_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(output_dir)
    
    print("File unzipped successfully!")

# Example usage
if __name__ == '__main__':
    # Set the path to the tgz file
    tgz_file_path = "../../audio_file/raw_imported/instrument_dataset.tgz"

    # Set the output directory where the contents will be extracted
    output_directory = "../../audio_file/raw_imported/instrument"

    # Unzip the tgz file
    unzip_tgz_file(tgz_file_path, output_directory)