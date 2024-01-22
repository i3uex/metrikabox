import json
import os


class ClassLoader:
    def get_class(self, audio_file:str, num_items:int) -> list:
        return ['']*num_items


class ClassLoaderFromFolderName(ClassLoader):
    def get_class(self, audio_file:str, num_items:int) -> list:
        base_path, class_name, file_name = audio_file.rsplit('/', 2)
        return [class_name]*num_items


class ClassLoaderFromDict(ClassLoader):

    def __init__(self, classes_dict):
        self.classes_dict = classes_dict

    def get_class(self, audio_file:str, num_items:int) -> list:
        base_path, class_name, file_name = audio_file.rsplit('/', 2)
        return [self.classes_dict[file_name]]*num_items


class ClassLoaderFromSameFileName(ClassLoader):
    def get_class(self, audio_file:str, num_items:int) -> list:
        json_file = audio_file.replace(".mp3", ".json")
        if not os.path.exists(json_file):
            print(f"{json_file} NOT found")
            return [None]*num_items
        with open(json_file) as f:
            classes = json.load(f)
        out_data = []
        for class_ in classes:
            out_data.extend([class_['value']] * (round(float(class_['to'])) - round(float(class_['from']))))
        if len(out_data) < num_items:
            out_data.append(out_data[-1])
        return out_data[:num_items]
