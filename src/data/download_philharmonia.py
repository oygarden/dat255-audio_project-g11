import os
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

from pydub import AudioSegment

from pydub.exceptions import CouldntDecodeError

def download_and_unzip(url, dir_name):
    if not os.path.isdir(dir_name):
        # Download the file
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc="Downloading Philharmonia dataset")
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

        # Navigate to the "all-samples" directory and extract sub-zip files
        all_samples_dir = Path(dir_name) / "all-samples"
        for instrument_zip in all_samples_dir.iterdir():
            if instrument_zip.is_file() and instrument_zip.suffix == '.zip':
                # Create a directory for the instrument
                instrument_dir = all_samples_dir / instrument_zip.stem
                instrument_dir.mkdir(exist_ok=True)
                # Unzip the instrument file into its directory
                with zipfile.ZipFile(instrument_zip, 'r') as zip_ref:
                    zip_ref.extractall(instrument_dir)
                os.remove(instrument_zip)

                # Now handle subdirectories within this instrument directory
                for sub_dir in instrument_dir.iterdir():
                    if sub_dir.is_dir():  # It's a subdirectory with samples
                        # Move all files from the subdirectory to the instrument directory
                        for sub_file in sub_dir.iterdir():
                            if sub_file.is_file():
                                shutil.move(str(sub_file), str(instrument_dir))
                        # Remove the now-empty subdirectory
                        os.rmdir(sub_dir)
        # Remove the __MACOSX directory if it exists
        macosx_dir = Path(dir_name) / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

        # Remove the zip file
        os.remove(zip_file_name)
        print("Downloaded and unzipped Philharmonia dataset")
    else:
        print("Philharmonia dataset already downloaded")

def download_philharmonia(directory):
    
    directory.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_PATH = directory / 'Philharmonia'
    
    url = "https://philharmonia-assets.s3-eu-west-1.amazonaws.com/uploads/2020/02/12112005/all-samples.zip"

    download_and_unzip(url, DOWNLOAD_PATH)
    
    base_dir = DOWNLOAD_PATH / "all-samples"
    
    # Check is metadata exists
    if os.path.exists(DOWNLOAD_PATH / 'metadata.csv'):
        print("Philharmonia metadata already exists")
        return pd.read_csv(DOWNLOAD_PATH / 'metadata.csv')
    
    # Create a metadata file
    data = []
    
    dir_to_label = {
        "banjo": "Banjo",
        "bass-clarinet": "Clarinet",
        "bassoon": "Bassoon",
        "cello": "Cello",
        "clarinet": "Clarinet",
        "contrabassoon": "Contrabassoon",
        "english-horn": "English_horn",
        "double-bass": "Double_bass",
        "flute": "Flute",
        "french-horn": "French_horn",
        "guitar": "Guitar",
        "mandolin": "Mandolin",
        "oboe": "Oboe",
        "percussion": "Percussion",
        "saxophone": "Saxophone",
        "trombone": "Trombone",
        "trumpet": "Trumpet",
        "tuba": "Tuba",
        "viola": "Viola",
        "violin": "Violin",
        # Percussion instruments (should generalize)
        "agogo-bells": "Percussive_Bells",
        "banana-shaker": "Shaker",
        "bass-drum": "Bass_drum",
        "bell-tree": "Percussive_Bells",
        "cabasa": "Percussion_Other",
        "castanets": "Percussion_Other",
        "Chinese-cymbal": "Cymbal",
        "Chinese-hand-cymbals": "Cymbal",
        "clash-cymbals": "Cymbal",
        "cowbell": "Cowbell",
        "djembe": "Drum",
        "djundjun": "Drum",
        "flexatone": "Percussion_Other",
        "guiro": "Percussion_Other",
        "lemon-shaker": "Shaker",
        "motor-horn": "Percussion_Other",
        "ratchet": "Percussion_Other",
        "sheeps-toenails": "Percussion_Other",
        "sizzle-cymbal": "Cymbal",
        "sleigh-bells": "Percussive_Bells",
        "snare-drum": "Snare_drum",
        "spring-coil": "Percussion_Other",
        "squeaker": "Percussion_Other",
        "strawberry-shaker": "Shaker",
        "surdo": "Drum",
        "suspended-cymbal": "Cymbal",
        "swanee-whistle": "Whistle",
        "tambourine": "Tambourine",
        "tam-tam": "Drum",
        "tenor-drum": "Drum",
        "Thai-gong": "Gong",
        "train-whistle": "Whistle",
        "triangle": "Triangle",
        "vibraslap": "Percussion_Other",
        "washboard": "Percussion_Other",
        "whip": "Percussion_Other",
        "wind-chimes": "Wind_chimes",
        "woodblock": "Woodblock", 
        "tom-toms": "Drum",
    }
    
    # Get the list of all files in base_dir that end with '.mp3'
    all_mp3_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_dir) for f in filenames if f.endswith('.mp3')]

    # Initialize the progress bar
    with tqdm(total=len(all_mp3_files), desc="Building Philharmonia metadata", unit="file") as pbar:
        for file_path in all_mp3_files:
            # Extract the base filename (e.g., 'saxophone_A3_1_forte_normal.mp3')
            base_filename = os.path.basename(file_path)
            try:
                # Extract label from filename
                label = base_filename.split('_')[0]
                
                label_map = dir_to_label.get(label)

                # Convert mp3 to wav
                audio = AudioSegment.from_mp3(file_path)
                wav_file = os.path.splitext(base_filename)[0] + '.wav'
                audio.export(os.path.join(os.path.dirname(file_path), wav_file), format='wav')

                # Add the metadata to the list
                data.append({
                    "fname": wav_file,
                    "path": os.path.join(os.path.dirname(file_path), wav_file),
                    "label": label_map,
                    "instrument": label,
                    "dataset": "Philharmonia"
                })

            # Handle corrupted files
            except CouldntDecodeError as e:
                print(f"Could not process {base_filename}")

            # Update the progress bar
            pbar.update(1)
    
    # Create a DataFrame from the metadata
    df = pd.DataFrame(data)
    
    # Save datafrae to CSV
    df.to_csv(DOWNLOAD_PATH / 'metadata.csv', index=False)
    
    return df # Return the metadata DataFrame
    
    
if __name__ == '__main__':
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    directory = PROJECT_ROOT / 'data' / 'external'
    download_philharmonia(directory)