import os
import json
import numpy as np
from multiprocessing import Lock
from typing import Union
import keras
from pydub import AudioSegment
from sklearn.preprocessing import LabelBinarizer
from audio_classifier.loaders import FileLoader
from audio_classifier.model.builder import NormLayer
from audio_classifier.utils import Singleton

BATCH_SIZE = os.environ.get('CLASSIFIER_BATCH_SIZE', '128')
try:
    BATCH_SIZE = int(BATCH_SIZE)
except ValueError:
    BATCH_SIZE = 128


class AudioModel(metaclass=Singleton):
    """
    Class to load a model and predict audio
    """
    def __init__(self, model: Union[keras.Model, str], model_config: Union[dict, str] = ""):
        """
        Class to load a model and predict audio
        :param model_id: ID of the model to load
        """
        self.mtx = Lock()
        if issubclass(type(model), keras.Model):
            self.model = model
        else:
            self.model = keras.models.load_model(model, compile=False, custom_objects={"NormLayer": NormLayer})
        if type(model_config) is dict:
            self.model_config = model_config
        else:
            with open(model_config) as f:
                self.model_config = json.load(f)
        self.encoder = LabelBinarizer().fit(self.model_config['classes'])

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
