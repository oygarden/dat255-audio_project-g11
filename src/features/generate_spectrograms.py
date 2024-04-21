import os
from pathlib import Path
import shutil
import imageio
import librosa
from matplotlib import cm
import numpy as np
import pandas as pd
from tqdm import tqdm

def clear_directory(folder_path):
    # Check if the directory exists
    if not os.path.exists(folder_path):
        print(f"The directory {folder_path} does not exist.")
        return
    
    # Loop through all items in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # If it's a file or symlink, delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If it's a directory, delete it and all its contents
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def generate_spectrograms(df, output_dir, fixed_length_seconds=3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    clear_directory(output_dir)
    
    # Sample rate
    sr = 44100  
    # Calculate fixed length in samples
    fixed_length_samples = int(fixed_length_seconds * sr)
    
    # Initialize an empty list to store spectrogram paths
    spectrogram_paths = []
    
    for index, row in tqdm(df.iterrows(), desc='Generating spectrograms', total=len(df)):
        # Load the audio file
        y, sr = librosa.load(row['path'], sr=sr, mono=True)
        
        if len(y) < fixed_length_samples:
            # Calculate the amount of silence needed
            padding_needed = fixed_length_samples - len(y)
            # Generate a random offset for the silence padding
            offset = np.random.randint(0, padding_needed)
            
            # Pad the audio signal with silence before and after based on the random offset
            silence_before = np.zeros(offset)
            silence_after = np.zeros(padding_needed - offset)
            y_padded = np.concatenate((silence_before, y, silence_after))
        else:
            # If the audio is longer than the fixed length, truncate it
            y_padded = y[:fixed_length_samples]
        
        # Generate the spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y_padded, sr=sr, n_mels=128, fmax=22000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize the spectrogram
        norm_log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        # Apply a colormap
        colored_spec = cm.viridis(norm_log_mel_spec)
        
        # Convert to RGB
        colored_spec_rgb = (colored_spec[..., :3] * 255).astype(np.uint8)

        # Define the file name and save path
        file_name = os.path.basename(row['path']).replace('.wav', '_spectrogram.png')
        save_path = os.path.join(output_dir, file_name)
        
        # Save the spectrogram image
        imageio.imwrite(save_path, colored_spec_rgb)
        
        # Append the save path to the list
        spectrogram_paths.append(save_path)
    
    # Add the list as a new column to the DataFrame
    df['spectrogram_path'] = spectrogram_paths

    return df  # Return the updated DataFrame

if __name__ == '__main__':
    # Get the directory containing this script
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Navigate to the project root
    project_root = script_dir.parent.parent
    
    SPEC_DIR = project_root / 'data' / 'processed' / 'mixed_audio_clips' /'spectrograms'
    
    metadata_path = project_root / 'data' / 'processed' / 'mixed_audio_clips' / 'mixed_clips_df.csv'
    
    # Check if metadata file exists
    if os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        
        # Check if 'spectrogram_path' column exists, if not, generate spectrograms
        if 'spectrogram_path' not in metadata.columns:
            metadata = generate_spectrograms(metadata, SPEC_DIR)
        
        # Save updated metadata
        metadata.to_csv(metadata_path, index=False)
    else:
        print(f"Metadata file not found at {metadata_path}, make sure to run the 'mix_audio_clips.py' script first.")