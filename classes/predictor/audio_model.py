import os
import json
import pickle
import numpy as np
from multiprocessing import Lock
from typing import Union
import tensorflow as tf
from pydub import AudioSegment
from loaders import FileLoader
from utils import Singleton
from config import MODEL_CONFIG_FOLDER, CHECKPOINTS_FOLDER

BATCH_SIZE = os.environ.get('CLASSIFIER_BATCH_SIZE', '128')
try:
    BATCH_SIZE = int(BATCH_SIZE)
except ValueError:
    BATCH_SIZE = 128


class AudioModel(metaclass=Singleton):
    """
    Class to load a model and predict audio
    """
    def __init__(self, model: Union[tf.keras.Model, str]):
        """
        Class to load a model and predict audio
        :param model_id: ID of the model to load
        """
        self.mtx = Lock()
        if issubclass(type(model), tf.keras.Model):
            self.model = model
        else:
            self.model = tf.keras.models.load_model(f'{CHECKPOINTS_FOLDER}/{model}', compile=False, custom_objects={"tf": tf})
        with open(f'{MODEL_CONFIG_FOLDER}/{model}/LabelEncoder.pkl', 'rb') as f:
            self.encoder = pickle.load(f)
        with open(f'{MODEL_CONFIG_FOLDER}/{model}/model-config.json') as f:
            self.model_config = json.load(f)

    def predict_without_format(self, audio: Union[str, np.ndarray, AudioSegment]) -> np.ndarray:
        """
        Predict the audio
        :param audio: Audio to predict
        :return: Prediction
        """
        return self.model.predict(FileLoader(self.model_config['sample_rate'], self.model_config['window'], self.model_config['step']).load(audio), batch_size=BATCH_SIZE)

    def predict(self, audio: Union[str, np.ndarray, AudioSegment]):
        """
        Predict the audio
        :param audio: Audio to predict
        :return: Prediction
        """
        return self.format_output(self.predict_without_format(audio))

    def format_output(self, y: np.ndarray):
        """
        Format the output
        :param y: Output
        :return: Formatted output
        """
        return y


if __name__ == '__main__':
    import sys
    print(AudioModel(sys.argv[2]).predict(sys.argv[1]))
