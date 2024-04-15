import librosa
import numpy as np
import os
import pandas as pd
import soundfile as sf

def mix_audio_clips(clip_paths, output_path, sr=44100):
    # Load the first clip
    mixed_clip, _ = librosa.load(clip_paths[0], sr=sr)
    
    # Load and mix each subsequent clip
    for clip_path in clip_paths[1:]:
        clip, _ = librosa.load(clip_path, sr=sr)
        # Ensure the clips are of the same length
        min_len = min(len(mixed_clip), len(clip))
        mixed_clip = mixed_clip[:min_len] + clip[:min_len]
        
    # Normalize the mixed clip to prevent clipping
    mixed_clip = mixed_clip / np.max(np.abs(mixed_clip))
    
    # Save the mixed clip to an output file
    sf.write(output_path, mixed_clip, sr)
    
    
# Example usage
path1 = '/home/bjorn/git/dat255-audio_project-g11/data/external/fsdkaggle2018/FSDKaggle2018.audio_train/0a0a8d4c.wav'
path2 = '/home/bjorn/git/dat255-audio_project-g11/data/external/fsdkaggle2018/FSDKaggle2018.audio_train/0a2a5c05.wav'
mix_audio_clips([path1, path2], 'mixed_clip.wav')

def determine_frequency_range(audio_path, sr=44100):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute the short-time Fourier transform (STFT)
    D = np.abs(librosa.stft(y))
    
    # Sum the spectral energy in each frequency range
    low_sum = D[:int(250 / sr * D.shape[0])].sum()
    mid_sum = D[int(250 / sr * D.shape[0]):int(2000 / sr * D.shape[0])].sum()
    high_sum = D[int(2000 / sr * D.shape[0]):].sum()
    
    # Determine the dominant frequency range
    if low_sum > mid_sum and low_sum > high_sum:
        return 'low'
    elif mid_sum > high_sum:
        return 'mid'
    else:
        return 'high'
    
    
# Example usage
audio_clips = [{'path': path1, 'label': 'violin'},
               {'path': path2, 'label': 'drum'}]

# Create the DataFrame
df = pd.DataFrame(audio_clips)

# Determine the frequency range for each clip and add it to the DataFrame
df['frequency_range'] = df['path'].apply(determine_frequency_range)

print(path1, determine_frequency_range(path1))
print(path2, determine_frequency_range(path2))

# Example: Filter clips by frequency range
low_freq_clips = df[df['frequency_range'] == 'low']
mid_freq_clips = df[df['frequency_range'] == 'mid']
high_freq_clips = df[df['frequency_range'] == 'high']

# Test frequency range determination with 10 random clips, and provide a player for each clip
import random
import IPython.display as ipd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

data_path = os.path.join(project_root,'data','external', 'fsdkaggle2018' ,'FSDKaggle2018.audio_train')

print(data_path)

# retrieve fsdkaggle metadata
metadata = pd.read_csv(os.path.join(project_root,'data','external', 'fsdkaggle2018' ,'FSDKaggle2018.meta', 'train_post_competition.csv'))

# priunt metadata columns
print(metadata.columns)

# add path column to metadata
metadata['path'] = data_path + '/' + metadata['fname']

# retrieve 10 random clips
df = metadata.sample(5)

# determine frequency range for each clip and add it to the DataFrame
df['frequency_range'] = df['path'].apply(determine_frequency_range)

# display the clips and their frequency range
import pygame
from time import sleep

# display the clips and their frequency range
for index, row in df.iterrows():
    print(row['path'], row['frequency_range'])
    
    # Initialize pygame mixer
    pygame.mixer.init()
    
    # Load the audio file
    pygame.mixer.music.load(row['path'])
    
    # Play the audio file
    pygame.mixer.music.play()
    
    # Wait for the audio file to finish playing
    while pygame.mixer.music.get_busy():
        sleep(1)