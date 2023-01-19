import os
import json
import pickle
from multiprocessing import Lock
from tensorflow.python.keras.models import load_model
from loaders import FileLoader
from utils import Singleton

BATCH_SIZE = os.environ.get('CLASSIFIER_BATCH_SIZE', '128')
try:
    BATCH_SIZE = int(BATCH_SIZE)
except ValueError:
    BATCH_SIZE = 128

class AudioModel(metaclass=Singleton):
    def __init__(self, model_id):
        self.mtx = Lock()
        self.model = load_model(f'checkpoints/{model_id}', compile=False)
        with open(f'LabelEncoder-{model_id}.pkl', 'rb') as f:
            self.encoder = pickle.load(f)
        with open(f'model-config-{model_id}.json') as f:
            self.model_config = json.load(f)

    def predict(self, audio):
        y = self.model.predict(FileLoader(self.model_config['sample_rate'], self.model_config['window'], self.model_config['step']).load(audio), batch_size=BATCH_SIZE)
        return self._format_output(y)

    def _format_output(self, y):
        return y

if __name__ == '__main__':
    import sys
    print(AudioModel(sys.argv[2]).predict(sys.argv[1]))