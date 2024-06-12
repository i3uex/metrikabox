import os
import json
import pickle
from multiprocessing import Lock
import tensorflow as tf
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
    def __init__(self, model_id):
        """
        Class to load a model and predict audio
        :param model_id: ID of the model to load
        """
        self.mtx = Lock()
        self.model = tf.keras.models.load_model(f'{CHECKPOINTS_FOLDER}/{model_id}', compile=False, custom_objects={"tf": tf})
        with open(f'{MODEL_CONFIG_FOLDER}/{model_id}/LabelEncoder.pkl', 'rb') as f:
            self.encoder = pickle.load(f)
        with open(f'{MODEL_CONFIG_FOLDER}/{model_id}/model-config.json') as f:
            self.model_config = json.load(f)

    def predict(self, audio):
        """
        Predict the audio
        :param audio: Audio to predict
        :return: Prediction
        """
        y = self.model.predict(FileLoader(self.model_config['sample_rate'], self.model_config['window'], self.model_config['step']).load(audio), batch_size=BATCH_SIZE)
        return self._format_output(y)

    def _format_output(self, y):
        """
        Format the output
        :param y: Output
        :return: Formatted output
        """
        return y


if __name__ == '__main__':
    import sys
    print(AudioModel(sys.argv[2]).predict(sys.argv[1]))
