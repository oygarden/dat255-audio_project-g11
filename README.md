DAT255 Audio Classification Project
==============================

Group members:
- Bjørn Egil Øygarden
- Lars Magnus Nordeide
- Bjørn-Sigurd Sælevik Fjose

Demo of the application can be found [here](https://www.youtube.com/watch?v=8xIL1kVeMj0)

# Deep learning audio classification

In this project we set out to train an AI model to detect the presence of musical instruments in sound clips. The data is converted into a spectrogram image showing frequencies over time and is fed into a visual learner to find the characteristics of each instrument.

![image](https://github.com/oygarden/dat255-audio_project-g11/assets/89018956/bf9a78f0-7686-425c-8d7d-11dfd0cda53b)

## Datasets

- [__Freesound Dataset Kaggle 2018 (FSDKaggle2018)__]()
    - A large dataset containing 41 catagories of recorded sounds, but only those containing musical instruments are used resulting in 21 categories.
- [__VocalSet: A Singing Voice Dataset__](https://zenodo.org/records/1193957)
    - Audio recordings of professional singers using a variety of vocal techniques in different musical contexts.
- [__Musical Instrument's Sound Dataset__](https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset/code)
    - A smaller dataset containing Guitar, Drums, Piano and Violin.
- [__IRMAS: a dataset for instrument recognition in musical audio signals__](https://www.upf.edu/web/mtg/irmas)
- [__Philharmonia__](https://philharmonia.co.uk/resources/sound-samples/)


Given that our model is made to identifying the instruments present in a segment of music, it was logical to develop a user interface that allows users to upload a song and receive ongoing feedback regarding the instruments being played.

We opted to implement a straightforward Flask server back-end. This setup facilitates our main processing tasks using Python methods that were honed during the experimental phase of our project. Below is a diagram illustrating the data flow:

![image](https://github.com/oygarden/dat255-audio_project-g11/assets/89018956/d9542eff-7ece-4dd9-a5d5-c57f068c62b1)

The client-side of the application consists of a basic webpage designed to manage song uploads. Interaction between the client and the server is managed through a bidirectional WebSocket connection established with FlaskSocketIO. Upon upload, the song is temporarily stored on the server, segmented into smaller pieces, and processed sequentially. For each segment, a spectrogram is generated and subsequently analyzed by our model to predict which instruments are audible in that specific clip.

As predictions are made, both the spectrograms and the associated multi-label predictions are transmitted to a buffer on the client side. Once the initial segments have been processed, playback of the song begins on the client. During playback, the application displays the predictions and spectrograms synchronously with the current audio segment, enhancing the user's interactive experience with real-time insights.

For the best experience, it is recommended to run the Flask server locally on your machine by running the `app.py` script. From project root: `python src/app/app.py`. The web-app can also be accessed [here](https://flask.onegard.no/). This deployment is in a simple minimal Docker container running on a small home server, so keep that in mind if the speeds are bit slow. 

The `app.py` script uses a model hosted in a huggingface repo, which can be found [here](https://huggingface.co/gruppe11/audio-classifier/tree/main). 

Screenshot of web client:
![image](https://github.com/oygarden/dat255-audio_project-g11/assets/89018956/6de51455-1958-41df-9310-cf4cd23f58c3)


# Local setup

Before running anything locally, make sure you install the required dependencies from `requirements.txt`, or, if only running the Flask app, `requirements-prod.txt` will suffice. This can be done by running `pip install -r requirements.txt`.

# Download the data

If you want to download the datasets to your machine, you can run the `download_data.py` script by doing `python src/data/download_data.py`. Keep in mind this will download about __21GB__ of audio files. After downloading, this data can be found in `data/external`, and then split in to their corresponding datasets, `fsdkaggle2018`, `IRMAS`, `MISD`, `Philharmonia` and `VocalSet`. There will also be generated a metadata file that contains labels and paths to all the datasets. Most of the datasets are retrieved using GET requests, except the `Musical Instruments Sound Datasets`, which uses the Kaggle API. The Kaggle API requires authentication, so before downloading, make sure you have a Kaggle account, and go into `Settings` and then under `API`, you can create a new Token. This will trigger a download of a `kaggle.json` file. This file you want to put in the `~/.kaggle` directory, which is where the API will look for credentials. 

# Create mixed clip dataset

Now, to train a model to predict instruments, you will need to train it on a multi-label dataset of overlapping instruments. To create this dataset, you can run the `mix_audio_clips.py` script. `python src/features/mix_audio_clips.py`. Before you run it, you might want to adjust how many clips you would like to generate. This variable `n_clips` can be found at the bottom of the script inside `if __name__ == '__main__':`. This script will first sort the whole dataset into _frequency ranges_, and then it will generate clips based on complimenting frequency ranges. This is to mimic realistic mixing and to avoid muddy mixes.  


# Generate spectrograms

After the mixed clips have been generated, you can run the `generate_spectrogram.py` script. `python src/features/generate_spectrograms.py`. This will go through the audio clips and generate a corresponding spectrogram and add the path to the csv file. 

# Training a model

Training a model can be done by running the `train_model.py` script. `python src/models/train_model.py`. Parameteres can be adjusted in the script. The model will be saved in `models` directory. If you wan to try this model with the Flask application, you can comment out the huggingface loader, and instead use the local loader. 


# Problems

- There were not many audio clips for bass and low frequency instruments.
- There were some trouble when having up to 45 different classes for multi-labeling, so the labels were generalized.

# Future

- It would be interesting to validate on real labeled music.
- Non-spectrogram representations of the audio.


# Conclusion

The project was very fun to work with, even though there were some problems along the way. Although the project diverted a bit from the initial project description, we found it quite interesting and motivating to follow up this idea we had of classifying instruments as a song was playing. One negative might be that the building of a dataset took up more time than initially thought, and it would be beneficial to spend some more time fine tuning the actual models used. 

----

# Project Description

### Introduction

Systems able to recognize sounds directly from audio recordings are widely applicable. In this project, you’ll attempt to create an audio tagging system by extracting audio clip image representations and then using computer vision-based classification models. You can consider constructing your system using a relatively large-scale competition data set and then evaluate it on its ability to recognize and distinguish more specialized sounds on locally generated recordings.

![image](https://github.com/oygarden/dat255-audio_project-g11/assets/89018956/94dde56e-82ef-4d63-8b65-615657c9713a)

### Goals

1. Investigate and construct models for automatic audio tagging of noisy recordings.
2. Adapt this to smaller data sets of audio recordings, either by using a setup motivated by your above findings or by transfer learning.
3. Construct an audio tagging application.

### Methods and materials

To achieve Goal 1 of the project, you can, for example, use the FSDKaggle2018 used in the Freesound General-Purpose Audio Tagging Challenge on Kaggle. There are 41 categories of audio clips, and the goal is to classify each clip. For the second objective, you can look for a data set on your own or construct one yourself. As part of the project, you should investigate ways to do data augmentation for audio. You’ll make use of a variety of Python audio libraries, e.g., librosa. You should also look into fastxtend, a library built on top of fastai. To construct the application, you’re free to use any solution you know or want to investigate. A natural starting point is the deployment solutions used in the fastai course. Consider not converting audio to images but instead setting up an audio classification framework that operates on audio representations of the data.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
