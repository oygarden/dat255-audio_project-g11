import os
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import requests

from tqdm import tqdm

def download_and_unzip(url, dir_name):
    if not os.path.isdir(dir_name) or not os.listdir(dir_name):
        # Download the file
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading VocalSet")
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
        
        print("Downloaded and unzipped VocalSet dataset")
    else: 
        print("VocalSet dataset already downloaded")
        

def download_vocalset(directory):
    
    directory.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH = directory / 'VocalSet'
    
    url = "https://zenodo.org/records/1193957/files/VocalSet.zip?download=1"
    
    download_and_unzip(url, DOWNLOAD_PATH)
    
    
    # Remove the __MACOSX directory if it exists
    macosx_dir = Path(DOWNLOAD_PATH) / "__MACOSX"
    if os.path.isdir(macosx_dir):
        shutil.rmtree(macosx_dir)
    
    base_dir = DOWNLOAD_PATH / "FULL"
    
    # Check is metadata exists
    if os.path.exists(DOWNLOAD_PATH / 'metadata.csv'):
        print("VocalSet metadata already exists")
        return pd.read_csv(DOWNLOAD_PATH / 'metadata.csv')
    
    # Create a metadata file
    data = []
    
    # Get the total number of .wav files
    total_files = sum([len(files) for r, d, files in os.walk(base_dir) if any(f.endswith('.wav') for f in files)])

    # Create a progress bar
    pbar = tqdm(total=total_files, desc="Building VocalSet metadata", unit="file")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".wav"):
                data.append({
                    "fname": file,
                    "path": os.path.join(root,file),
                    "label": "Vocal",
                    "dataset": "VocalSet"
                })
                
                # Update progress bar
                pbar.update(1)
                
    pbar.close()
                    
    df = pd.DataFrame(data)
    
    # Save the metadata
    df.to_csv(DOWNLOAD_PATH / 'metadata.csv', index=False)
    
    return df
    
if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    directory = PROJECT_ROOT / 'data' / 'external'
    download_vocalset(directory)