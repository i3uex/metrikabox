import math
from typing import Union
from pydub import AudioSegment
from librosa.util import buf_to_float
import numpy as np
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass

def __window(a, window=DEFAULT_WINDOW, step=DEFAULT_STEP, sr=DEFAULT_SAMPLE_RATE) -> np.ndarray:
    w = int(window * sr)
    o = int(step * sr)
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    return np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]

def apply_window(audio: Union[AudioSegment, np.array], window=DEFAULT_WINDOW, step=DEFAULT_STEP, sr=DEFAULT_SAMPLE_RATE):
    if type(audio) is str:
        audio = AudioSegment.from_file(audio).set_frame_rate(sr).set_channels(1)
    if type(audio) is AudioSegment:
        audio = buf_to_float(audio.get_array_of_samples(), n_bytes=audio.sample_width)
    return np.expand_dims(__window(np.pad(audio, math.ceil(sr/2*window), mode="symmetric"), window=window, step=step, sr=sr), 2)

def get_mels_from_hop_and_win_lengths(hop_length, win_length, input_size=DEFAULT_SAMPLE_RATE*DEFAULT_WINDOW):
    return int(math.floor((input_size - win_length) / hop_length) + 1)

