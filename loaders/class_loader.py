class ClassLoader:
    def get_class(self, audio_file, num_items):
        return ['']*num_items


class ClassLoaderFromFolderName(ClassLoader):
    def get_class(self, audio_file, num_items):
        base_path, class_name, file_name = audio_file.rsplit('/', 2)
        return [class_name]*num_items


class ClassLoaderFromDict(ClassLoader):

    def __init__(self, classes_dict):
        self.classes_dict = classes_dict

    def get_class(self, audio_file, num_items):
        base_path, class_name, file_name = audio_file.rsplit('/', 2)
        return [self.classes_dict[file_name]]*num_items
