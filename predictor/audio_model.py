import os
import pickle
from abc import abstractmethod
from multiprocessing import Lock
from tensorflow.python.keras.models import load_model
from utils import apply_window, SingletonABCMeta
from config import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP


BATCH_SIZE = os.environ.get("CLASSIFIER_BATCH_SIZE", "256")
try:
    BATCH_SIZE = int(BATCH_SIZE)
except ValueError:
    BATCH_SIZE = 256

class AudioModel(metaclass=SingletonABCMeta):
    def __init__(self, model_id, sample_rate=SAMPLE_RATE, window=CONTEXT_WINDOW, step=PROCESSING_STEP):
        model_id = int(model_id)
        self.mtx = Lock()
        self.model = load_model("checkpoints/%d" % model_id, compile=False)
        with open("LabelEncoder-%d.pkl" % model_id, "rb") as f:
            self.encoder = pickle.load(f)
        self.sr, self.window, self.step = sample_rate, window, step

    def predict(self, audio):
        y = self.model.predict(apply_window(audio, window=self.window, step=self.step, sr=self.sr), batch_size=BATCH_SIZE)
        return self._format_output(y)

    @abstractmethod
    def _format_output(self, y):
        pass

if __name__ == '__main__':
    import sys
    print(AudioModel(sys.argv[2]).predict(sys.argv[1]))