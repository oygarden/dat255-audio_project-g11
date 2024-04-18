import os
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import requests

from tqdm import tqdm

def download_and_unzip(url, dir_name):
    if not os.path.isdir(dir_name):
        # Download the file
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        zip_file_name = f"{dir_name}.zip"
        with open(zip_file_name, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

        # Unzip the file
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(dir_name)

        # Remove the zip file
        os.remove(zip_file_name)
        

def download_vocalset(directory):
    
    directory.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH = directory / 'VocalSet'
    
    url = "https://zenodo.org/records/1193957/files/VocalSet.zip?download=1"
    
    download_and_unzip(url, DOWNLOAD_PATH)
    
    print("Downloaded and unzipped VocalSet dataset")
    
    # Remove the __MACOSX directory if it exists
    macosx_dir = Path(DOWNLOAD_PATH) / "__MACOSX"
    if os.path.isdir(macosx_dir):
        shutil.rmtree(macosx_dir)
    
    base_dir = DOWNLOAD_PATH / "FULL"
    
    # Create a metadata file
    data = []
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                data.append({
                    "fname": file,
                    "path": os.path.join(root,file),
                    "label": "Vocal"
                })
                    
    df = pd.DataFrame(data)
    
    return df
    
if __name__ == '__main__':
    download_vocalset()