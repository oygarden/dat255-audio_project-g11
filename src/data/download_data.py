import os

import pandas as pd
from src.data.download_fsdkaggle import download_fsdkaggle
from src.data.download_vocalset import download_vocalset
import concurrent.futures

def download_all(directory):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        fsdkaggle_meta_future = executor.submit(download_fsdkaggle, directory)
        vocalset_meta_future = executor.submit(download_vocalset, directory)

    fsdkaggle_meta = fsdkaggle_meta_future.result()
    vocalset_meta = vocalset_meta_future.result()
    
    # Combine the metadata
    metadata = pd.concat([fsdkaggle_meta, vocalset_meta], ignore_index=True)

    return metadata

if __name__ == '__main__':
    # Get the path to the current file
    current_file_path = os.path.realpath(__file__)
    
    print(current_file_path)

    # Get the path to the project root directory
    project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    print(project_root_dir)

    # Append the 'data/external' path to the project root directory
    directory = os.path.join(project_root_dir, 'data')
    
    print(directory)

    download_all(directory)