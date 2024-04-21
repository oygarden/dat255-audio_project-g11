from pathlib import Path
import random
import shutil
import numpy as np
import os
import pandas as pd

import librosa
import soundfile as sf
import colorednoise as cn

from tqdm import tqdm

from src.data.download_data import download_all

def determine_frequency_range(audio_path, sr=44100):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute the short-time Fourier transform (STFT)
    D = np.abs(librosa.stft(y))
    
    # Define the frequency ranges
    ranges = {
        'sub_bass': (20, 60),
        'bass': (60, 250),
        'low_midrange': (250, 500),
        'midrange': (500, 2000),
        'upper_midrange': (2000, 4000),
        'presence': (4000, 6000),
        'brilliance': (6000, 20000),
    }
    
    # Compute the spectral energy in each range
    energy = {name: D[int(low / sr * D.shape[0]):int(high / sr * D.shape[0])].sum()
              for name, (low, high) in ranges.items()}
    
    # Determine the dominant frequency range
    return max(energy, key=energy.get)

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
    
# Function to map labels in your DataFrame
def map_instrument_labels(row, instrument_map):
    labels = row['label'].split(', ')
    generalized_labels = set(instrument_map[label] for label in labels if label in instrument_map)
    return ', '.join(generalized_labels)

def generate_mixed_audio_clips(df, output_folder, n_clips, sr=44100, clip_length=3):
    
    instruments = ['Hi-hat', 'Saxophone', 'Trumpet' ,'Glockenspiel' ,'Cello', 'Clarinet',
                 'Snare_drum', 'Oboe' ,'Flute', 'Chime' ,'Bass_drum', 'Harmonica', 'Gong',
                'Double_bass', 'Tambourine' ,'Cowbell' ,'Electric_piano', 'Acoustic_guitar',
                'Violin_or_fiddle' ,'Finger_snapping', 'Vocal' ,'Guitar' ,'Drums', 'Piano',
                'Organ', 'Electric_guitar' ,'Tuba', 'Bassoon', 'Drum', 'Percussion_Other',
                'Percussive_Bells', 'Shaker' ,'Cymbal', 'Whistle' ,'Triangle', 'Wind_chimes',
                'Woodblock', 'French_horn', 'Trombone', 'Mandolin' ,'Contrabassoon',
                'English_horn' ,'Violin' ,'Viola', 'Banjo'
                ]
    
    
    genre_instruments = {
        'jazz': ['Saxophone', 'Trumpet', 'Double_bass', 'Clarinet', 'Trombone', 'Snare_drum', 'Bass_drum', 'Piano', 'Electric_piano'],
        'classical': ['Violin', 'Viola', 'Cello', 'Double_bass', 'Flute', 'Oboe', 'Clarinet', 'Bassoon', 'Contrabassoon', 'English_horn', 'French_horn', 'Trombone', 'Tuba', 'Organ', 'Piano', 'Glockenspiel', 'Percussive_Bells'],
        'rock': ['Electric_guitar', 'Drums', 'Bass_drum', 'Snare_drum', 'Electric_piano', 'Acoustic_guitar', 'Piano'],
        'blues': ['Harmonica', 'Acoustic_guitar', 'Electric_guitar', 'Piano', 'Drums'],
        'folk': ['Acoustic_guitar', 'Banjo', 'Mandolin', 'Violin_or_fiddle', 'Harmonica'],
        'electronic': ['Electric_piano', 'Synthesizer', 'Drum_machine'],
        'world': ['Shaker', 'Gong', 'Wind_chimes', 'Woodblock', 'Triangle', 'Tambourine', 'Drums', 'Percussion_Other'],
        'wildcard':instruments,
        'pop': ['Vocal', 'Electric_guitar', 'Acoustic_guitar', 'Piano', 'Electric_piano', 'Synthesizer', 'Drums', 'Bass_drum', 'Snare_drum', 'Finger_snapping', 'Guitar']
    }

    
    # Map instruments to more general categories
    instrument_map = {  # Not sure if were actually using this, but keeping it here for now
        'Snare_drum': 'Snare_drum', 'Bass_drum': 'Bass_drum', 'Hi-hat': 'Hi-hat',
        'Tambourine': 'Tambourine', 'Gong': 'Gong', 'Cowbell': 'Cowbell',
        'Violin_or_fiddle': 'Violin_or_fiddle', 'Cello': 'Cello', 'Double_bass': 'Bass',
        'Acoustic_guitar': 'Acoustic_guitar', 'Guitar': 'Guitar','Electric_guitar': 'Electric_guitar','Electric_piano': 'Electric_piano', 'Flute': 'Flute',
        'Clarinet': 'Clarinet', 'Oboe': 'Oboe', 'Saxophone': 'Saxophone', 
        'Trumpet': 'Trumpet', 'Trombone': 'Trombone', 'Harmonica': 'Harmonica',
        'Glockenspiel': 'Glockenspiel', 'Chime': 'Chime',
        'Electric_piano': 'Electric_piano', 'Synthesizers': 'Synthesizers', 'Drum_machine': 'Drums',
        'Finger_snapping': 'Finger_snapping', 'Vocal': 'Vocal', 'Piano':'Piano', 'Organ':'Organ', 'Drums':'Drums'
    }

    #df['generalized_label'] = df.apply(map_instrument_labels(instrument_map), axis=1)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    elif os.listdir(output_folder):
        clear_directory(output_folder)

    mixed_clips_info = []

    # Wrap the range function with tqdm to display a progress bar
    for i in tqdm(range(n_clips), desc="Generating mixed clips"):
        labels, genre, output_path = mix_clips_from_different_ranges(df, genre_instruments, f"{output_folder}/mixed_clip_{i}", sr, clip_length)
        if labels and output_path:
            mixed_clips_info.append({'path': output_path, 'labels': ', '.join(labels), 'genre': genre})

    return pd.DataFrame(mixed_clips_info)

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

def adjust_for_genre(clip, genre, sr=44100):
    if genre == 'classical':
        # Increase dynamic range; classical music often has wide dynamic swings
        clip = clip * (1 + np.var(clip))
    elif genre == 'rock':
        # Apply compression to decrease dynamic range; rock music often has a compressed, upfront sound
        clip = librosa.effects.preemphasis(clip)
    elif genre == 'jazz':
        # Slightly increase dynamic range and add a subtle reverb to emulate live jazz environments
        clip = clip * (1 + 0.5 * np.var(clip))
        clip = librosa.effects.preemphasis(clip, coef=0.97)  # Slight pre-emphasis
    elif genre == 'blues':
        # Apply mild compression and a warmer tone by reducing high frequencies
        clip = librosa.effects.preemphasis(clip, coef=0.95)
    elif genre == 'folk':
        # Enhance the natural dynamics and apply a gentle high-pass filter to emulate acoustic settings
        clip = librosa.effects.hpss(clip)[1]
    elif genre == 'electronic':
        # Normalize to ensure uniform loudness levels typical of electronic music
        clip = librosa.util.normalize(clip)
    elif genre == 'world':
        # Apply a slight increase in dynamic range and a moderate reverb to reflect diverse instrumentation and spaces
        clip = clip * (1 + 0.3 * np.var(clip))
    elif genre == 'wildcard':
        # For wildcard, randomly choose to apply one of the effects mildly to not bias the genre
        effects = [lambda x: x, lambda x: librosa.effects.preemphasis(x, coef=0.98), lambda x: x * (1 + 0.2 * np.var(x)), librosa.util.normalize]
        clip = random.choice(effects)(clip)
    elif genre == 'pop':
        clip = librosa.util.normalize(clip)
        clip = librosa.effects.preemphasis(clip, coef=0.98)
        clip = librosa.effects.harmonic(clip) * 0.1 + clip
        clip = librosa.effects.preemphasis(clip, coef=0.97)

    return clip


def add_noise(clip, noise_type='white', snr=20):
    if noise_type == 'white':
        # Generate white noise
        noise = np.random.normal(0, 1, len(clip))
    elif noise_type == 'pink':
        # Generate pink noise using colorednoise
        # The exponent for pink noise is 1, beta = 1
        noise = cn.powerlaw_psd_gaussian(1, len(clip))
    elif noise_type == 'brownian':
        # Generate brownian noise using colorednoise
        # The exponent for brownian noise is 2, beta = 2
        noise = cn.powerlaw_psd_gaussian(2, len(clip))
    else:
        # TODO: Load custom noise file?
        return clip
    
    
    sig_power = np.sum(clip ** 2) / len(clip)
    noise_power = np.sum(noise ** 2) / len(noise)
    scale = (sig_power / noise_power) / (10 ** (snr / 10))
    noise = noise * np.sqrt(scale)
    
    return clip + noise

def random_slice_reassemble(audio_segment, num_slices=4):
    slice_length = len(audio_segment) // num_slices
    slices = [audio_segment[i * slice_length:(i + 1) * slice_length] for i in range(num_slices)]
    random.shuffle(slices)
    return sum(slices)

def vary_speed(clip, sr, min_speed=0.9, max_speed=1.1):
    speed_factor = random.uniform(min_speed, max_speed)
    return librosa.effects.time_stretch(clip, rate=speed_factor)

def get_random_clip(full_clip, sr, clip_length, silence_threshold=0.01, max_attempts=10):
    # If the full clip is shorter than the desired length, return the full clip
    if len(full_clip) <= sr * clip_length:
        return full_clip

    for _ in range(max_attempts):
        start = random.randint(0, len(full_clip) - sr * clip_length)
        clip = full_clip[start : start + sr * clip_length]
        
        # If the maximum absolute value in the clip is above the threshold, return the clip
        if np.max(np.abs(clip)) > silence_threshold:
            return clip
    
    # If no non-silent clip was found after max_attempts, return None
    return None

def mix_clips_from_different_ranges(df, genre_instruments, output_file_name, sr=44100, clip_length=3, min_groups=3, max_groups=8):
    # Randomly select a genre
    genre = random.choice(list(genre_instruments.keys()))
    instruments = genre_instruments[genre]

    # Filter df for the selected instruments
    df_genre = df[df['label'].isin(instruments)]

    grouped = df_genre.groupby('frequency_range')  # Group the clips by frequency range
    num_groups = random.randint(min_groups, max_groups)
    selected_groups = random.sample(list(grouped.groups), min(num_groups, len(grouped.groups)))

    selected_clips = []
    used_instruments = set()
    for group in selected_groups:
        group_df = grouped.get_group(group)
        group_df = group_df[~group_df['label'].isin(used_instruments)]  # Exclude used instruments
        if not group_df.empty:
            selected_clip = group_df.sample(1).iloc[0]
            selected_clips.append(selected_clip)
            used_instruments.add(selected_clip['label'])

    mixed_clip = np.zeros(int(sr * clip_length))

    for row in selected_clips:
        full_clip, _ = librosa.load(row['path'], sr=sr)
    
        # Randomly select a non-silent portion of the clip
        clip = get_random_clip(full_clip, sr, clip_length)
        if clip is None:
            continue
        
        clip = adjust_for_genre(clip, genre)  # Add genre-specific adjustments
        
        # Randomly vary the speed of the clip
        if random.random() < 0.5:
            clip = vary_speed(clip, sr=sr)
        
        #Randomly slice and reassemble the clip
        if random.random() < 0.3:
            clip = random_slice_reassemble(clip)
        
        # Randomly add noise
        if random.random() < 0.5:  # Chance of adding noise
            noise_types = ['white', 'pink', 'custom', 'brownian']
            noise_type = random.choice(noise_types)  # Randomly select the type of noise to add
            clip = add_noise(clip, noise_type=noise_type)
        
        if len(clip) < len(mixed_clip):
            clip = np.tile(clip, int(np.ceil(len(mixed_clip) / len(clip))))[:len(mixed_clip)]
        mixed_clip += clip[:len(mixed_clip)]
        
    mixed_clip = mixed_clip / np.max(np.abs(mixed_clip))
    output_path = f"{output_file_name}.wav"
    sf.write(output_path, mixed_clip, sr)

    return [row['label'] for row in selected_clips], genre, output_path

if __name__ == '__main__':
    
    # Get the directory containing this script
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    # Navigate to the project root
    project_root = script_dir.parent.parent

    DATA_DIR = project_root / 'data' / 'external'
    
    # read metadata
    meta = pd.read_csv(DATA_DIR / 'metadata.csv')

    # Check if 'frequency_range' column exists, if not, make it
    if 'frequency_range' not in meta.columns:
        tqdm.pandas(desc="Determining frequency ranges")
        meta.loc[:, 'frequency_range'] = meta['path'].progress_apply(determine_frequency_range)
        
        # save updated metadata
        meta.to_csv(DATA_DIR / 'metadata.csv', index=False)
        
    

    PATH = project_root / 'data' / 'processed' / 'mixed_audio_clips'
    
    n_clips = 30000
    
    mixed_clips_df = generate_mixed_audio_clips(meta, PATH, n_clips)
    
    # Save mixed_clips_df to a CSV file
    mixed_clips_df.to_csv(os.path.join(PATH, 'mixed_clips_df.csv'), index=False)
    
    
    