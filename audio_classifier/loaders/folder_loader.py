import os
import glob
import random
import multiprocessing
from typing import Tuple, Collection
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
from audio_classifier import constants
from audio_classifier.utils import LOGGER
from audio_classifier.loaders import BaseLoader, FileLoader
from audio_classifier.loaders.class_loader import ClassLoader

BASE_PATH = ''


class FolderLoader:
    def __init__(
            self,
            file_loader: BaseLoader = None,
            class_loader: ClassLoader = ClassLoader(),
            audio_formats: Collection[str] = (".wav", ".mp3")
    ):
        """
        Class to load audio files from a folder
        :param sample_rate: Desired sample rate
        :param window: Length of the window in seconds
        :param step: Length of the step in seconds
        :param class_loader: Class loader to use
        """
        self.Y = []
        self.X = []
        self.class_loader = class_loader
        if not file_loader:
            file_loader = FileLoader(
                sample_rate=constants.DEFAULT_SAMPLE_RATE,
                window=constants.DEFAULT_WINDOW,
                step=constants.DEFAULT_STEP
            )
            print("File loader not provided. Using default file loader")
        self.file_loader = file_loader
        self.audio_formats = audio_formats

    def load(
            self,
            folder: str,
            max_files: int = None,
            classes2avoid: Collection[str] = (),
    ) -> Tuple[Collection[np.ndarray], Collection[str]]:
        """
        Load audio files from a folder
        :param folder: Folder containing the audio files
        :param max_files: Maximum number of files to load
        :param classes2avoid: Audio classes to avoid from training
        :param audio_formats: Desired audio formats
        :return: Array with the audio data and a list with the classes
        """
        if self.X or self.Y:
            return self.X, self.Y
        if not folder.endswith("/"):
            folder += "/"
        items = list(
            filter(
                lambda item: not os.path.isdir(item) and (
                    any([item.lower().endswith(f) for f in self.audio_formats]) if self.audio_formats else True
                ),
                glob.glob(folder + '**', recursive=True)
            )
        )
        if max_files:
            items = items[:max_files]
        # Load audio files
        num_processes = multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            futures = [ex.submit(self.file_loader.load, audio_file) for audio_file in items]
            with tqdm(total=len(futures), desc='Loading files') as pbar:
                # Process loaded audios
                for i, (af, future) in enumerate(zip(items, futures)):
                    try:
                        x = future.result()
                    except Exception:
                        LOGGER.exception(f'Error in file {items[i]}. Skipping')
                        continue
                    y = self.class_loader.get_class(af, x.shape[0])
                    filtered_data = list(filter(lambda _item: _item[1] not in classes2avoid, zip(x, y)))
                    if filtered_data:
                        x, y = list(zip(*filtered_data))
                        self.X.extend(x)
                        self.Y.extend(y)
                    pbar.update()
        LOGGER.info("Shuffling dataset")
        list2shuffle = list(zip(self.X, self.Y))
        random.Random().shuffle(list2shuffle)
        self.X, self.Y = zip(*list2shuffle)
        return self.X, self.Y
