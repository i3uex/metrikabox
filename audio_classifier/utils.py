from typing import Union
from pydub import AudioSegment
import numpy as np
from audio_classifier.config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP
import logging

LOGGER = logging.getLogger("audio_classifier")


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


def __window(
        x: np.ndarray,
        window_seconds: float = DEFAULT_WINDOW,
        step_seconds: float = DEFAULT_STEP,
        sr: int = DEFAULT_SAMPLE_RATE
) -> np.ndarray:
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


def load_audio(audio: Union[str, np.ndarray, AudioSegment], sr=16000) -> np.ndarray:
    if type(audio) is str:
        audio = AudioSegment.from_file(audio)
    if type(audio) is AudioSegment:
        audio = audio.set_channels(1).set_frame_rate(sr).set_sample_width(2).get_array_of_samples()
    return audio


def apply_window(
        audio: np.ndarray,
        window: float = DEFAULT_WINDOW,
        step: float = DEFAULT_STEP,
        sr: int = DEFAULT_SAMPLE_RATE,
        pad_mode="constant"
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
    items_per_step = int(sr * step)
    spare_items = len(audio) % items_per_step
    pad_width = (items_per_step - spare_items) if spare_items > 0 else 0
    return __window(
        np.pad(audio, pad_width=(0, pad_width), mode=pad_mode),
        window_seconds=window,
        step_seconds=step,
        sr=sr
    )
