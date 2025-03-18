import importlib
from inspect import getmembers, isfunction, isclass

CUSTOM_MODELS = {f"custom.{model_name}": model for model_name, model in getmembers(importlib.import_module('audio_classifier.model.classification'), isfunction)}
AVAILABLE_MODELS = {f"keras.{model_name}": model for model_name, model in getmembers(importlib.import_module('tensorflow.keras.applications'), isfunction)}
AVAILABLE_MODELS.update(CUSTOM_MODELS)
AVAILABLE_KERAS_OPTIMIZERS = {optimizer_name: optimizer for optimizer_name, optimizer in getmembers(importlib.import_module('tensorflow.keras.optimizers'), isclass)}
AVAILABLE_CLASS_LOADERS = {class_loader_name: class_loader for class_loader_name, class_loader in getmembers(importlib.import_module('audio_classifier.loaders.class_loader'), isclass)}
AVAILABLE_AUDIO_AUGMENTATIONS = {augmentation_name: augmentation for augmentation_name, augmentation in getmembers(importlib.import_module('audio_classifier.augmentations.audio'), isclass)}
AVAILABLE_SPECTROGRAM_AUGMENTATIONS = {augmentation_name: augmentation for augmentation_name, augmentation in getmembers(importlib.import_module('audio_classifier.augmentations.spectrogram'), isclass)}