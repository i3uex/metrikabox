import logging
from typing import Union
import soxr
import numpy as np
from pydub import AudioSegment
from audio_classifier import constants

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
        window_seconds: float = constants.DEFAULT_WINDOW,
        step_seconds: float = constants.DEFAULT_STEP,
        sr: int = constants.DEFAULT_SAMPLE_RATE
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


def load_audio(audio: Union[str, np.ndarray, AudioSegment], sr: int = 16000, max_duration: float = None) -> np.ndarray:
    if type(audio) is str:
        audio = AudioSegment.from_file(audio, duration=max_duration)
    if type(audio) is AudioSegment:
        frame_rate = audio.frame_rate
        audio = audio.set_channels(1).set_sample_width(2).get_array_of_samples()
        if frame_rate != sr:
            audio = soxr.resample(audio, frame_rate, sr)
    return audio


def apply_window(
        audio: np.ndarray,
        window: float = constants.DEFAULT_WINDOW,
        step: float = constants.DEFAULT_STEP,
        sr: int = constants.DEFAULT_SAMPLE_RATE,
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
