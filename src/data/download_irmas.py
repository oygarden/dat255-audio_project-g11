

import os
from pathlib import Path
import zipfile
import pandas as pd
import requests
from tqdm import tqdm


def download_and_unzip(url, dir_name):
    if not os.path.isdir(dir_name):
        # Download the file
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading IRMAS")
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
        print("Downloaded and unzipped IRMAS dataset")
    else: 
        print("IRMAS dataset already downloaded")

def download_irmas(directory):
    directory.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH = directory / 'IRMAS'
    
    url = "https://zenodo.org/records/1290750/files/IRMAS-TrainingData.zip?download=1"
    
    download_and_unzip(url, DOWNLOAD_PATH)
    
    base_dir = DOWNLOAD_PATH / "IRMAS-TrainingData"
    
    # Check is metadata exists
    if os.path.exists(DOWNLOAD_PATH / 'metadata.csv'):
        print("IRMAS metadata already exists")
        return pd.read_csv(DOWNLOAD_PATH / 'metadata.csv')
    
    # Create a metadata file
    data = []
    
    dir_to_label = {
        "cel": "Cello",
        "cla": "Clarinet",
        "flu": "Flute",
        "gac": "Acoustic_guitar",
        "gel": "Electric_guitar",
        "org": "Organ",
        "pia": "Piano",
        "sax": "Saxophone",
        "tru": "Trumpet",
        "vio": "Violin_or_fiddle",
        "voi": "Vocal"
    }
    
    # Get the total number of .wav files
    total_files = sum([len(files) for r, d, files in os.walk(base_dir) if any(f.endswith('.wav') for f in files)])

    # Create a progress bar
    pbar = tqdm(total=total_files, desc="Building IRMAS metadata", unit="file")

    # Iterate over the directories and use the directory name as the label
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                
                dir_name = os.path.basename(root)
                
                label = dir_to_label.get(dir_name)
                
                data.append({
                    "fname": file,
                    "path": os.path.join(root, file),
                    "label": label,
                    "dataset": "IRMAS"
                })

                # Update the progress bar
                pbar.update(1)

    # Close the progress bar
    pbar.close()
    
    df = pd.DataFrame(data)
    
    # Save the metadata
    df.to_csv(DOWNLOAD_PATH / 'metadata.csv', index=False)
    
    return df
    
    
    
if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    directory = PROJECT_ROOT / 'data' / 'external'
    download_irmas(directory)