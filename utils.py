import math
from typing import Union
from pydub import AudioSegment
from librosa.util import buf_to_float
import numpy as np
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP


class Singleton(type):
    """
    Singleton metaclass
    """
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


def __window(x: np.array, window_seconds=DEFAULT_WINDOW, step_seconds=DEFAULT_STEP,
             sr=DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """
    Apply a window to the audio
    :param x: Audio in np array format
    :param window_seconds: seconds of the window
    :param step_seconds: seconds of the step
    :param sr: sample rate of the audio
    :return: windowed audio
    """
    window_frames = int(window_seconds * sr)
    step_frames = int(step_seconds * sr)
    shape = (x.size - window_frames + 1, window_frames)
    strides = x.strides * 2  # * window ??
    return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape, writeable=False)[0::step_frames]


def apply_window(audio: Union[str, np.array, AudioSegment],
                 window: float = DEFAULT_WINDOW,
                 step: float = DEFAULT_STEP,
                 sr: int = DEFAULT_SAMPLE_RATE,
                 pad_mode="symmetric"
                 ) -> np.ndarray:
    """
    Apply a window to the audio
    :param audio: Audio in np array format
    :param window: length of the window in seconds
    :param step: length of the step in seconds
    :param sr: sample rate of the audio
    :param pad_mode: padding mode for the np array
    :param dtype: desired output type. Default is int16
    :return: windowed audio
    """
    if type(audio) is str:
        audio = AudioSegment.from_file(audio)
    if type(audio) is AudioSegment:
        audio = audio.set_frame_rate(sr).set_channels(1).set_sample_width(2).get_array_of_samples()
    return np.expand_dims(
        __window(np.pad(audio, math.ceil(sr / 2 * window), mode=pad_mode), window_seconds=window, step_seconds=step, sr=sr), 2)


def get_mels_from_hop_and_win_lengths(hop_length: int, win_length,
                                      input_size=DEFAULT_SAMPLE_RATE * DEFAULT_WINDOW) -> int:
    """
    Get the number of mels from the hop and win lengths
    :param hop_length: hop length
    :param win_length: window length
    :param input_size: input size
    :return:
    """
    return int(math.floor((input_size - win_length) / hop_length) + 1)
