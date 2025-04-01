from collections import Counter
from typing import Collection
from audio_classifier import constants
from audio_classifier.utils import LOGGER
from audio_classifier.loaders import ClassLoaderFromFolderName


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
        self.model_builder = None

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

    def get_model_builder(self, model_config):
        if self.model_builder is None:
            raise NotImplementedError("Model builder not implemented")
        else:
            return self.model_builder(**model_config)
