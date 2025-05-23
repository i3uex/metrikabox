import json
import os
from typing import Collection, Optional

from audio_classifier.utils import LOGGER


class ClassLoader:
    """
    Base class to load the classes of an audio file
    """
    def get_class(self, audio_file:str, num_items:int) -> Collection[Optional[str]]:
        return ['']*num_items


class ClassLoaderFromFolderName(ClassLoader):
    """
    Class to load the classes of an audio file from the folder name
    """
    def get_class(self, audio_file: str, num_items: int) -> Collection[Optional[str]]:
        base_path, class_name, file_name = audio_file.rsplit('/', 2)
        return [class_name]*num_items


class ClassLoaderFromDict(ClassLoader):
    """
    Class to load the classes of an audio file from a given dictionary
    """
    def __init__(self, classes_dict):
        self.classes_dict = classes_dict

    def get_class(self, audio_file: str, num_items: int) -> Collection[Optional[str]]:
        base_path, class_name, file_name = audio_file.rsplit('/', 2)
        return [self.classes_dict[file_name]]*num_items


class ClassLoaderFromSameFileName(ClassLoader):
    """
    Class to load the classes of an audio file from the same file name in JSON extension
    """
    def get_class(self, audio_file: str, num_items: int) -> Collection[Optional[str]]:
        json_file = audio_file.replace(".mp3", ".json")
        if not os.path.exists(json_file):
            LOGGER.warning(f"{json_file} NOT found. Using None as class.")
            return [None]*num_items
        with open(json_file) as f:
            classes = json.load(f)
        out_data = []
        for class_ in classes:
            out_data.extend([class_['value']] * (round(float(class_['to'])) - round(float(class_['from']))))
        if len(out_data) < num_items:
            out_data.append(out_data[-1])
        return out_data[:num_items]
