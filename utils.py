import math
from typing import Union

from pydub import AudioSegment
from pydub.effects import normalize
from librosa.util import buf_to_float
import numpy as np
import random

from config import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP

random.seed(42)

MAX_ITEMS = 10000
INPUT_SHAPE = (SAMPLE_RATE*CONTEXT_WINDOW, 1)

def _window(a, window=CONTEXT_WINDOW, step=PROCESSING_STEP, sr=SAMPLE_RATE) -> np.ndarray:
    w = int(window * sr)
    o = int(step * sr)
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    return np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]

def apply_window(audio: Union[AudioSegment, np.array], window=CONTEXT_WINDOW, step=PROCESSING_STEP, sr=SAMPLE_RATE):
    if type(audio) is AudioSegment:
        audio = buf_to_float(audio.get_array_of_samples(), n_bytes=audio.sample_width)

    return np.expand_dims(_window(np.pad(audio, math.ceil(sr/2*window), mode="symmetric"), window=window, step=step, sr=sr), 2)

def prepare_audio(audio):
    audio = AudioSegment.from_file(audio).set_frame_rate(SAMPLE_RATE).set_channels(1)
    return apply_window(audio)