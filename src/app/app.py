from flask import Flask, request, render_template, send_from_directory, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import shutil
from pydub import AudioSegment
import random
import librosa
import numpy as np
import imageio
from matplotlib import cm

from fastai.vision.all import *

from huggingface_hub import hf_hub_url, hf_hub_download
from fastai.learner import load_learner


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
# Get the system path to the current file
current_file_path = os.path.realpath(__file__)

# Get the system path to the project directory
project_dir = os.path.dirname(os.path.dirname(current_file_path))

# Set the upload folder to be the system path to the project + /src/app/uploads
upload_folder = os.path.join(project_dir, 'app', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
print("Upload folder:", app.config['UPLOAD_FOLDER'])

def ensure_dir_exists(dir_path):
    """Ensure that a directory exists, create it if it doesn't."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
ensure_dir_exists(upload_folder)
    
socketio = SocketIO(app, logger=True, engineio_logger=True, max_http_buffer_size=1e8)

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'flac', 'ogg', 'm4a'}

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format="wav")
    return wav_path

@socketio.on('song_uploaded')
def handle_song_upload(message):
    print("Received song upload message")
    
    upload_folder = app.config['UPLOAD_FOLDER']

    # Clear existing files in the upload folder
    clear_directory(upload_folder)

    # Proceed with saving the new song
    filename = secure_filename(message['filename'])
    file_path = os.path.join(upload_folder, filename)
    save_file(file_path, message['song_data'])
    
    # Convert to wav if necessary
    if not filename.rsplit('.', 1)[1].lower() == 'wav':
        file_path = convert_to_wav(file_path)
        filename = filename.rsplit('.', 1)[0] + '.wav'
    
    segment_length = 3 # seconds
    split_song(file_path, segment_length * 1000)
    
    song_url = request.host_url + 'songs/' + filename
    emit('song_ready', {'song_url': song_url})
    
    static_path = os.path.join(upload_folder, '..','static')
    
    ensure_dir_exists(static_path)
    
    generate_and_predict_spectrograms(os.path.join(upload_folder, 'segments'), static_path)
    
def save_file(file_path, song_data):
    print("Saving file to", file_path)
    # Ensure song_data is a bytes-like object
    if isinstance(song_data, bytes):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(song_data)
    else:
        print("Error: song_data is not in bytes format")

@app.route('/songs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def split_song(file_path, segment_length_ms):
    segment_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'segments')
    if not os.path.isdir(segment_dir):
        print("Creating directory", segment_dir)
        os.makedirs(segment_dir, exist_ok=True)
        
    song = AudioSegment.from_wav(file_path)
    for i in range(0, len(song), segment_length_ms):
        segment = song[i:i + segment_length_ms]
        print("Exporting segment", i // segment_length_ms)
        segment.export(os.path.join(segment_dir, f"segment_{i // segment_length_ms}.wav"), format="wav")
        

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def generate_and_predict_spectrograms(audio_dir, output_dir, sr=44100):
    # Clear the output directory at the start of each call
    clear_directory(output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    # Sort the audio files by the segment number
    audio_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

    for file in audio_files:
        # Assuming the filename format is "segment_x.wav"
        segment_index = file.split('_')[1].split('.')[0]

        audio_path = os.path.join(audio_dir, file)
        y, sr = librosa.load(audio_path, sr=sr, mono=True)

        # Generate spectrogram for the audio segment
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=22000)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        norm_log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        colored_spec = cm.viridis(norm_log_mel_spec)
        colored_spec_rgb = (colored_spec[..., :3] * 255).astype(np.uint8)

        # Save spectrogram
        spec_filename = f"segment_{segment_index}_spectrogram.png"
        save_path = os.path.join(output_dir, spec_filename)
        imageio.imwrite(save_path, colored_spec_rgb)

        # Make a mock prediction on the spectrogram
        #prediction = mock_predict_on_segment(save_path)
        
        prediction = predict_on_segment(save_path)
        
        spectrogram_url = url_for('static', filename=spec_filename)
        
        # Emit prediction to the client
        emit('prediction_ready', {'index': segment_index, 'prediction': prediction, 'spectrogram_url': spectrogram_url})


def mock_predict_on_segment(segment_path):
    # Add a short delay to simulate processing time
    import time
    time.sleep(1)
    
    # Mock prediction logic as before, adapted for a single segment
    print(f"Predicting on segment: {segment_path}")
    num_predictions = random.randint(1, 10)
    prediction = [random.choice(['drums', 'guitar', 'bass', 'vocals', 'synth', 'violin', 'tuba']) for _ in range(num_predictions)]
    print("Prediction:", prediction)
    return prediction

def get_x(r): 
    return r

def get_y(r):
    return null

def predict_on_segment(segment_path):
    
    # Make a prediction
    pred, _, probs = learn.predict(segment_path)
    
    # Apply a threshold to convert probabilities to binary predictions
    threshold = 0.1
    binary_preds = (probs > threshold).numpy()

    # Get the list of possible classes from the learner's data loaders
    class_names = learn.dls.vocab

    # Filter class names based on the binary predictions
    predicted_labels = [class_names[i] for i in range(len(class_names)) if binary_preds[i]]
    
    print("Predicted labels:", predicted_labels)
    return predicted_labels

# Load the pre-trained model

# local
#model_path = os.path.join(project_dir,'..','models', 'instrument_classifier3.pkl')
#learn = load_learner(model_path)


# Huggingface
REPO = "gruppe11/audio-classifier"
FILENAME = "instrument_classifier3.pkl"

model_url = hf_hub_url(REPO, FILENAME)

model_path = hf_hub_download(REPO, FILENAME)

learn = load_learner(model_path)

print("Model loaded")

if __name__ == '__main__':
    socketio.run(app, debug=True)
