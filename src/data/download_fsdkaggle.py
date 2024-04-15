import os
from pathlib import Path
import pandas as pd
import requests
import zipfile

def download_and_unzip(url, dir_name):
    if not os.path.isdir(dir_name):
        # Download the file
        response = requests.get(url)
        zip_file_name = f"{dir_name}.zip"
        with open(zip_file_name, 'wb') as f:
            f.write(response.content)

        # Unzip the file
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(dir_name)

        # Remove the zip file
        os.remove(zip_file_name)
        
        
def download_fsdkaggle(directory):
    
    DATA_DIR = Path(directory)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    SUB_DIRS = ['external','interim','processed','raw']    

    for d in SUB_DIRS:
        dir_path = DATA_DIR / d
        dir_path.mkdir(parents=True, exist_ok=True)
        
    DATA_DIR

    DOWNLOAD_PATH = DATA_DIR / 'external' / 'fsdkaggle2018'
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

    download_and_unzip('https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_test.zip', DOWNLOAD_PATH / 'FSDKaggle2018.audio_test')
    download_and_unzip('https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip', DOWNLOAD_PATH / 'FSDKaggle2018.audio_train')
    download_and_unzip('https://zenodo.org/records/2552860/files/FSDKaggle2018.meta.zip', DOWNLOAD_PATH / 'FSDKaggle2018.meta')
    
    print("Downloaded and unzipped FSDKaggle2018 dataset")
    
    return pd.read_csv(DOWNLOAD_PATH / 'FSDKaggle2018.meta' / 'train_post_competition.csv') # Return the metadata

if __name__ == '__main__':
    download_fsdkaggle()