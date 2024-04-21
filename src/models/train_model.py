import os
from pathlib import Path

import pandas as pd

from fastai.vision.all import *
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def check_for_df(directory):
    if os.path.isfile(directory / 'mixed_clips_df.csv'):
        return pd.read_csv(directory / 'mixed_clips_df.csv')
    else:
        print("No mixed_clips_df.csv file found in the directory. Please run the data preprocessing scripts first.")
        return None
    
def get_y(r): 
    return r['labels'].split(', ')

def get_x(r): 
    return r['spectrogram_path']

    
def train_model(metadata, directory, model_name):
    # Create a DataBlock
    db = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                splitter=RandomSplitter(seed=420),
                get_x=get_x,
                get_y=get_y,
                item_tfms=[Resize(224)],  # Resize all images to 224x224
                )

    dls = db.dataloaders(metadata, bs=64)
    
    learn = vision_learner(dls, 
                           resnet34, 
                           metrics=partial(accuracy_multi, thresh=0.5), 
                           loss_func=BCEWithLogitsLoss()
                           )
    
    lr = learn.lr_find()
    print("Learning rate finder results(lr.valley):" + str(lr.valley))
    
    learn.fit_one_cycle(5, lr.valley,cbs=[SaveModelCallback(), EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=3)])

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create a file name based on the model name and the timestamp
    model_file_name = f'{model_name}_{timestamp}.pkl'
    
    save_model(learn, directory / model_file_name)
    
    return learn

    
def save_model(learn, path):
    learn.export(path)
    print("Model saved successfully. Path: " + str(path))

    
if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    meta_directory = PROJECT_ROOT / 'data' / 'processed' / 'mixed_audio_clips'
    
    metadata = check_for_df(meta_directory)
    
    if metadata is not None:
        print("Data preprocessing script has been run successfully. You can now train the model.")

        directory = PROJECT_ROOT / 'models'
        
        model_name = 'resnet34'
        
        learn = train_model(metadata,directory, model_name)
        
    else:
        print("Please run the data preprocessing script before training the model.")