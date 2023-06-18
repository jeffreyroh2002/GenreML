import os
import zipfile


def unzip_file(file_path, extract_dir):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("File unzipped successfully.")


def main():
    zip_file = "IRMAS-TestingData-Part3.zip"
    extract_dir = "../../audio_file/raw_imported/irmas_instrument/testing"
    unzip_file(zip_file, extract_dir)

if __name__ == "__main__":
    main()
