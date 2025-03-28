from abc import abstractmethod
from collections import Counter
from typing import Collection, Type
import tensorflow as tf
from audio_classifier import constants
from audio_classifier.loaders import FolderLoader
from audio_classifier.loaders.class_loaders import ClassLoaderFromFolderName
from audio_classifier.loaders.data_loaders import EncodecLoader, AudioLoader
from audio_classifier.model import AudioModelBuilder
from audio_classifier.model.builder import EncodecModelBuilder, ModelBuilder
from audio_classifier.utils import LOGGER


class Dataset:

    def __init__(
            self,
            folder: str = None,
            window: float = constants.DEFAULT_WINDOW,
            step: float = constants.DEFAULT_STEP,
            classes2avoid: Collection[str] = (),
            class_loader=None,
            **kwargs
    ):
        if not folder:
            print("Error: No folder provided. Exiting")
            return
        self.folder = folder
        self.window = window
        self.step = step
        self.classes2avoid = classes2avoid
        if class_loader:
            self.class_loader = constants.AVAILABLE_CLASS_LOADERS[class_loader]()
        else:
            self.class_loader = ClassLoaderFromFolderName()
        self.data_loader = None

    def get_config(self):
        return {
            "folder": self.folder,
            "window": self.window,
            "step": self.step,
            "classes2avoid": self.classes2avoid,
            "class_loader": type(self.class_loader).__name__,
            "file_loader": type(self.data_loader.file_loader).__name__
        }

    def load(self):
        """
        Loads the data to train the model
        :return: loaded data in a tuple (x, y, num_classes)
        """
        if not self.data_loader:
            raise NotImplementedError("Data loader not implemented")
        x, y = self.data_loader.load(self.folder, classes2avoid=self.classes2avoid)
        assert len(y) == len(x)
        LOGGER.info(f"Number of items per class: {Counter(y)}")
        return x, y

    @abstractmethod
    def get_output_signature(self):
        pass

    @abstractmethod
    def get_model_builder(self):
        pass


class AudioDataset(Dataset):
    def __init__(
            self,
            sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
            stft_nfft: int = constants.DEFAULT_STFT_N_FFT,
            stft_win: int = constants.DEFAULT_STFT_WIN,
            stft_hop: int = constants.DEFAULT_STFT_HOP,
            stft_nmels: int = constants.DEFAULT_N_MELS,
            mel_f_min: int = constants.DEFAULT_MEL_F_MIN,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.stft_nfft = stft_nfft
        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.stft_nmels = stft_nmels
        self.mel_f_min = mel_f_min
        self.file_loader = AudioLoader(sample_rate=sample_rate, window=self.window, step=self.step)
        self.data_loader = FolderLoader(
            self.file_loader,
            class_loader=self.class_loader
        )
        self.model_builder = AudioModelBuilder

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "stft_nfft": self.stft_nfft,
            "stft_win": self.stft_win,
            "stft_hop": self.stft_hop,
            "stft_nmels": self.stft_nmels,
            "mel_f_min": self.mel_f_min
        })
        return config

    def get_output_signature(self):
        return tf.TensorSpec(shape=(int(self.sample_rate * self.window),), dtype=tf.int16)

    def get_model_builder(self):
        return AudioModelBuilder


class EncodecDataset(Dataset):
    def __init__(
            self,
            model: str = 'encodec_24khz',
            decode: bool = True,
            bandwidth: int = 6.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.file_loader = EncodecLoader(model=model, decode=decode, bandwidth=bandwidth)
        self.data_loader = FolderLoader(
            self.file_loader,
            class_loader=self.class_loader,
            audio_formats=['.ecdc']
        )
        self.model_builder = EncodecModelBuilder

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.file_loader.model_name,
            "decode": self.file_loader.decode,
            "expected_codebooks": self.file_loader.codebooks
        })
        return config

    def get_output_signature(self):
        return tf.TensorSpec(shape=(
            self.file_loader.frame_rate * self.window,
            128 if self.file_loader.decode else self.file_loader.codebooks,
        ), dtype=tf.float32)

    def get_model_builder(self):
        return EncodecModelBuilder

