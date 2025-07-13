import os
import shutil
import gdown
import zipfile


# Script to download datasets used in samformer (https://github.com/romilbert/samformer).
# Not tested on MACOS


# Function to download dataset from Google Drive
def download_dataset(url, output):
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        print(f"gdown command failed: {e}")
        print(
            "Please download the files manually from the provided link and place them into the correct directories."
        )
        exit(1)


# Function to unzip the downloaded file
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


# Function to find all .csv files and copy them to the destination directory
def copy_csv_files(src_dir, dest_dir):
    for root, _, files in os.walk(src_dir):
        if "__MACOSX" in root:
            continue
        for file in files:
            if file.endswith(".csv"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src_file, dest_file)


# Function to clean up the temporary directory and the downloaded zip file
def cleanup(temp_dir, zip_file):
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.isfile(zip_file):
        os.remove(zip_file)


# Main script execution
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)

    # Define directories
    data_dir = os.path.join(SCRIPT_DIR, "../data/samformer_datasets/")
    temp_dir = os.path.join(SCRIPT_DIR, "temp_dir")
    zip_file = os.path.join(SCRIPT_DIR, "all_six_datasets.zip")

    # Download dataset from Google Drive
    download_dataset(
        "https://drive.google.com/uc?id=1alE33S1GmP5wACMXaLu50rDIoVzBM4ik", zip_file
    )

    # Unzip the downloaded zip file
    unzip_file(zip_file, temp_dir)

    # Find all .csv files in the extracted files and copy them to the destination directory
    copy_csv_files(temp_dir, data_dir)

    # Clean up the temporary directory and the downloaded zip file
    cleanup(temp_dir, zip_file)
