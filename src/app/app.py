from flask import Flask, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../../data/raw'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('home'))
        
@app.route('/play/<filename>')
def play_audio(filename):
    return '''
    <audio controls>
        <source src="{}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <div id="spectrogram"></div>
    '''.format(url_for('uploaded_file', filename=filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)