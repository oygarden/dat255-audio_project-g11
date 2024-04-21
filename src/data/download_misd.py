import os
from pathlib import Path
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd

def download_misd(directory):
    
    DOWNLOAD_PATH = directory / 'MISD'
    
    try:
        api = KaggleApi()
        api.authenticate()
    except OSError as e:
        print("Error during authentication. Please make sure the kaggle.json file is located in the ~/.kaggle directory and contains valid credentials.")
        return
    except Exception as e:
        print("An unexpected error occurred during authentication:", e)
        return
    
    if not os.path.isdir(DOWNLOAD_PATH) or not os.listdir(DOWNLOAD_PATH):
        
        os.makedirs(DOWNLOAD_PATH, exist_ok=True)
        api.dataset_download_files('soumendraprasad/musical-instruments-sound-dataset', path=DOWNLOAD_PATH, unzip=True)

        print("Downloaded and unzipped MISD dataset")
        
    else:
        print("MISD dataset already downloaded")
        
    train_meta_path = DOWNLOAD_PATH / 'Metadata_Train.csv'
    test_meta_path = DOWNLOAD_PATH / 'Metadata_Test.csv'
    
    train_meta = pd.read_csv(train_meta_path)
    test_meta = pd.read_csv(test_meta_path)
    
    # Change column name "File" to "fname"
    train_meta.rename(columns={'FileName': 'fname'}, inplace=True)
    test_meta.rename(columns={'FileName': 'fname'}, inplace=True)
    
    # Change column name "Class"    to "label"
    train_meta.rename(columns={'Class': 'label'}, inplace=True)
    test_meta.rename(columns={'Class': 'label'}, inplace=True)
    
    # Add path column
    train_meta['path'] = DOWNLOAD_PATH / 'Train_submission' / 'Train_submission' / train_meta['fname']
    test_meta['path'] = DOWNLOAD_PATH / 'Test_submission' / 'Test_submission' / test_meta['fname']
    
    # Combine train and test metadata
    metadata = pd.concat([train_meta, test_meta], ignore_index=True)

    # Sound_Drum to Drums
    metadata['label'] = metadata['label'].replace('Sound_Drum', 'Drums')
    
    # Sound_Guitar to Guitar
    metadata['label'] = metadata['label'].replace(['Sound_Guitar', 'Sound_Guiatr'], 'Guitar')
    
    # Sound_Piano to Piano
    metadata['label'] = metadata['label'].replace('Sound_Piano', 'Piano')
    
    # Sound violin to Violin_or_fiddle
    metadata['label'] = metadata['label'].replace('Sound_Violin', 'Violin_or_fiddle')
    
    # Add dataset column
    metadata['dataset'] = 'MISD'
    
    return metadata

if __name__ == '__main__':
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    directory = PROJECT_ROOT / 'data' / 'external'
    download_misd(directory)