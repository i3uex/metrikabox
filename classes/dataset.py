from collections import Counter
from typing import Collection

from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP
from constants import AVAILABLE_CLASS_LOADERS
from loaders import ClassLoaderFromFolderName, FolderLoader


class Dataset:

    def __init__(
            self,
            folder: str,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            window: float = DEFAULT_WINDOW,
            step: float = DEFAULT_STEP,
            classes2avoid: Collection[str] = (),
            class_loader=None
    ):
        self.sample_rate = sample_rate
        self.window = window
        self.step = step
        self.folder = folder
        self.classes2avoid = classes2avoid
        if class_loader:
            class_loader = AVAILABLE_CLASS_LOADERS[class_loader]()
        else:
            class_loader = ClassLoaderFromFolderName()
        self.data_loader = FolderLoader(
            sample_rate=self.sample_rate,
            window=self.window,
            step=self.step,
            class_loader=class_loader
        )

    def load(self):
        """
        Loads the data to train the model
        :return: loaded data in a tuple (x, y, num_classes)
        """
        x, y = self.data_loader.load(self.folder, classes2avoid=self.classes2avoid)
        assert len(y) == len(x)
        print("Number of items per class:", Counter(y))
        return x, y