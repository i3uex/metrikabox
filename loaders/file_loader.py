from typing import Union
import numpy as np
from pydub import AudioSegment
from utils import apply_window

class FileLoader:
    def __init__(self, sample_rate, window, step):
        self.sr = sample_rate
        self.window = window
        self.step = step

    def load(self, audio: Union[AudioSegment, np.array]):
        return apply_window(audio, window=self.window, step=self.step, sr=self.sr)

