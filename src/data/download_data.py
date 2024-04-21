import os
from pathlib import Path

import pandas as pd
from src.data.download_fsdkaggle import download_fsdkaggle
from src.data.download_vocalset import download_vocalset
from src.data.download_misd import download_misd
from src.data.download_irmas import download_irmas
from src.data.download_philharmonia import download_philharmonia
import concurrent.futures

def download_all(directory):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fsdkaggle_meta_future = executor.submit(download_fsdkaggle, directory)
        vocalset_meta_future = executor.submit(download_vocalset, directory)
        misd_meta_future = executor.submit(download_misd, directory)
        irmas_meta_future = executor.submit(download_irmas, directory)
        philharmonia_meta_future = executor.submit(download_philharmonia, directory)

    fsdkaggle_meta = fsdkaggle_meta_future.result()
    vocalset_meta = vocalset_meta_future.result()
    misd_meta = misd_meta_future.result()
    irmas_meta = irmas_meta_future.result()
    philharmonia_meta = philharmonia_meta_future.result()
    
    # Combine the metadata
    metadata = pd.concat([fsdkaggle_meta, vocalset_meta, misd_meta, irmas_meta, philharmonia_meta], ignore_index=True)

    # Save metadata
    metadata.to_csv(directory / 'metadata.csv', index=False)
    
    return metadata

if __name__ == '__main__':
    # Get the path to the current file
    CURRENT_FILE_PATH = os.path.realpath(__file__)

    # Get the path to the project root directory
    PROJECT_ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))))

    # Append the 'data/external' path to the project root directory
    directory = PROJECT_ROOT_DIR / 'data' / 'external'

    download_all(directory)