from abc import abstractmethod
from collections import Counter
from typing import Collection
import tensorflow as tf
from audio_classifier import constants
from audio_classifier.loaders import ClassLoaderFromFolderName, FolderLoader, FileLoader
from audio_classifier.loaders.encodec_loader import EncodecLoader
from audio_classifier.utils import LOGGER


class Dataset:

    def __init__(
            self,
            folder: str = None,
            window: float = constants.DEFAULT_WINDOW,
            step: float = constants.DEFAULT_STEP,
            classes2avoid: Collection[str] = (),
            class_loader=None
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
            "sample_rate": self.folder,
            "window": self.window,
            "step": self.step,
            "classes2avoid": self.classes2avoid,
            "class_loader": type(self.class_loader).__name__
        }

    def load(self):
        """
        Loads the data to train the model
        :return: loaded data in a tuple (x, y, num_classes)
        """
        if not self.data_loader:
            print("Error: No data loader provided. Exiting")
            return ()
        x, y = self.data_loader.load(self.folder, classes2avoid=self.classes2avoid)
        assert len(y) == len(x)
        LOGGER.info(f"Number of items per class: {Counter(y)}")
        return x, y

    @abstractmethod
    def get_output_signature(self):
        pass


class AudioDataset(Dataset):
    def __init__(
            self,
            sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.file_loader = FileLoader(sample_rate=sample_rate, window=self.window, step=self.step)
        self.data_loader = FolderLoader(
            self.file_loader,
            class_loader=self.class_loader
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate
        })
        return config

    def get_output_signature(self):
        return tf.TensorSpec(shape=(int(self.sample_rate * self.window),), dtype=tf.int16)


class EncodecDataset(Dataset):
    def __init__(
            self,
            model: str = 'encodec_24khz',
            decode: bool = True,
            expected_codebooks: int = 8,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.file_loader = EncodecLoader(model=model, decode=decode, expected_codebooks=expected_codebooks)
        self.data_loader = FolderLoader(
            self.file_loader,
            class_loader=self.class_loader,
            audio_formats=['.ecdc']
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.file_loader.model_name,
            "decode": self.file_loader.decode,
            "expected_codebooks": self.file_loader.expected_codebooks
        })
        return config

    def get_output_signature(self):
        return tf.TensorSpec(shape=(
            (75 if self.file_loader.model_name == 'encodec_24khz' else 150) * self.window,
            128 if self.file_loader.decode else self.file_loader.expected_codebooks,
        ), dtype=tf.float32)
