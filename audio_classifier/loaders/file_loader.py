from typing import Union
import soxr
import numpy as np
from pydub import AudioSegment
from audio_classifier import constants
from audio_classifier.loaders import BaseLoader


class FileLoader(BaseLoader):
    """
    Class to load audio files
    """
    def __init__(self, sample_rate: int = constants.DEFAULT_SAMPLE_RATE, **kwargs):
        self.sr = sample_rate
        super().__init__(**kwargs)

    def __window(
            self,
            x: np.ndarray
    ) -> np.ndarray:
        """
        Apply a window to the audio
        :param x: Audio in np array format
        :param window_seconds: seconds of the window
        :param step_seconds: seconds of the step
        :param sr: sample rate of the audio
        :return: windowed audio
        """
        window_frames = int(self.window * self.sr)
        step_frames = int(self.step * self.sr)
        shape = (x.size - window_frames + 1, window_frames)
        strides = x.strides * 2  # * window ??
        return np.lib.stride_tricks.as_strided(x, strides=strides, shape=shape, writeable=False)[0::step_frames]

    def apply_window(
            self,
            audio: np.ndarray,
            pad_mode="constant"
    ) -> np.ndarray:
        """
        Apply a window to the audio
        :param audio: Audio in np array format
        :param pad_mode: padding mode for the np array
        :param dtype: desired output type. Default is int16
        :return: windowed audio
        """
        items_per_step = int(self.sr * self.step)
        spare_items = len(audio) % items_per_step
        pad_width = (items_per_step - spare_items) if spare_items > 0 else 0
        return self.__window(
            np.pad(audio, pad_width=(0, pad_width), mode=pad_mode)
        )

    def load_audio(self, audio: Union[str, np.ndarray, AudioSegment], max_duration: float = None) -> np.ndarray:
        if type(audio) is str:
            audio = AudioSegment.from_file(audio, duration=max_duration)
        if type(audio) is AudioSegment:
            frame_rate = audio.frame_rate
            audio = audio.set_channels(1).set_sample_width(2).get_array_of_samples()
            if frame_rate != self.sr:
                audio = soxr.resample(audio, frame_rate, self.sr)
        return audio

    def load(self, audio: Union[str, np.ndarray, AudioSegment], max_duration: float = None) -> np.ndarray:
        return self.apply_window(self.load_audio(audio, max_duration=max_duration))

