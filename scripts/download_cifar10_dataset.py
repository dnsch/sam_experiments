import os
import shutil
import tarfile
import urllib.request


# Function to download the CIFAR-10 dataset
def download_cifar10(url, output_path):
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        exit(1)


# Function to extract a .tar.gz file
def extract_tar_gz(tar_path, extract_to):
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"Extracted to {extract_to}")
    except Exception as e:
        print(f"Extraction failed: {e}")
        exit(1)


# Clean up downloaded archive
def cleanup(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        print(f"Removed {file_path}")


# Main script execution
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SCRIPT_DIR)

    # Define target paths
    data_dir = os.path.join(SCRIPT_DIR, "../data/cifar10_datasets/")
    os.makedirs(data_dir, exist_ok=True)

    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_file_path = os.path.join(SCRIPT_DIR, "cifar-10-python.tar.gz")

    # Download and extract
    download_cifar10(cifar_url, tar_file_path)
    extract_tar_gz(tar_file_path, data_dir)

    # Clean up tar file
    cleanup(tar_file_path)
