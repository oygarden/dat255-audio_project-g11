import os
from pathlib import Path
import pandas as pd
import requests
import zipfile
from tqdm import tqdm

def download_and_unzip(url, dir_name):
    if not os.path.isdir(dir_name):
        # Download the file
        response = requests.get(url)
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
    else:
        print("FSDKaggle2018 already downloaded, skipping download")
        
        
def download_fsdkaggle(directory):

    DOWNLOAD_PATH = directory / 'fsdkaggle2018'
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

    download_and_unzip('https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_test.zip', DOWNLOAD_PATH / 'FSDKaggle2018.audio_test')
    download_and_unzip('https://zenodo.org/records/2552860/files/FSDKaggle2018.audio_train.zip', DOWNLOAD_PATH / 'FSDKaggle2018.audio_train')
    download_and_unzip('https://zenodo.org/records/2552860/files/FSDKaggle2018.meta.zip', DOWNLOAD_PATH / 'FSDKaggle2018.meta')
    
    print("Downloaded and unzipped FSDKaggle2018 dataset")
    
    metadata = pd.read_csv(DOWNLOAD_PATH / 'FSDKaggle2018.meta' / 'train_post_competition.csv')
    
    # Remove license column
    metadata = metadata.drop(columns=['license'])
    
    # Remove freesound_id column
    metadata = metadata.drop(columns=['freesound_id'])
    
    # Remove "manually_verified" column
    metadata = metadata.drop(columns=['manually_verified'])
    
    # Add "path" column
    metadata['path'] = metadata['fname'].apply(lambda x: str(DOWNLOAD_PATH / 'FSDKaggle2018.audio_train' / x))
    
    print(metadata.head())
     
    return metadata

if __name__ == '__main__':
    download_fsdkaggle()