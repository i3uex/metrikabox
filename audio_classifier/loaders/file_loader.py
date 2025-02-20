from typing import Union
import numpy as np
from pydub import AudioSegment
from audio_classifier.utils import apply_window, load_audio


class FileLoader:
    """
    Class to load audio files
    """
    def __init__(self, sample_rate: int, window: float, step: float):
        self.sr = sample_rate
        self.window = window
        self.step = step

    def load(self, audio: Union[str, np.ndarray, AudioSegment], max_duration: float = None) -> np.ndarray:
        return apply_window(load_audio(audio, sr=self.sr, max_duration=max_duration), window=self.window, step=self.step, sr=self.sr)

