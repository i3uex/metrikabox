import importlib
from inspect import getmembers, isfunction, isclass

# Configuration items
DEFAULT_WINDOW = 2  # Context (in seconds) for the audio processing
DEFAULT_STEP = 1  # Every how many seconds the audio processing will be done
DEFAULT_CLASS_LOADER = "ClassLoaderFromFolderName"
DEFAULT_DATASET_TYPE = "Audio"

# Training hyperparams
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 250
DEFAULT_LR = 0.001
DEFAULT_EARLY_STOPPING_PATIENCE = 25
DEFAULT_REDUCE_LR_ON_PLATEAU_PATIENCE = 10
DEFAULT_OPTIMIZER = "Adam"
DEFAULT_CHECKPOINT_METRIC = "val_loss"
DEFAULT_EARLY_STOPPING_METRIC = "val_accuracy"
DEFAULT_MODEL_ID = "model_id"
CHECKPOINTS_FOLDER = "checkpoints"
MODEL_CONFIG_FOLDER = "model_config"

# STFT configuration
# Tested for 1s step and 2s win @ 16000 Hz
DEFAULT_SAMPLE_RATE = 16000  # Sample rate the audio will be converted to when training/predicting
DEFAULT_STFT_HOP = 256
DEFAULT_STFT_N_FFT = DEFAULT_STFT_HOP * 4
DEFAULT_STFT_WIN = DEFAULT_STFT_HOP * 4
DEFAULT_MEL_F_MIN = 0
DEFAULT_N_MELS = 128

# EnCodec configuration
DEFAULT_ENCODEC_MODEL = "encodec_24khz"
DEFAULT_ENCODEC_DECODE = True
DEFAULT_ENCODEC_BANDWIDTH = 6.

# Available items configuration
CUSTOM_MODELS = {f"custom.{model_name}": model for model_name, model in getmembers(importlib.import_module('audio_classifier.model.classification'), isfunction)}
AVAILABLE_MODELS = {f"keras.{model_name}": model for model_name, model in getmembers(importlib.import_module('tensorflow.keras.applications'), isfunction)}
AVAILABLE_MODELS.update(CUSTOM_MODELS)
AVAILABLE_KERAS_OPTIMIZERS = {optimizer_name: optimizer for optimizer_name, optimizer in getmembers(importlib.import_module('tensorflow.keras.optimizers'), isclass)}
AVAILABLE_CLASS_LOADERS = {class_loader_name: class_loader for class_loader_name, class_loader in getmembers(importlib.import_module('audio_classifier.loaders.class_loaders'), isclass)}
AVAILABLE_AUDIO_AUGMENTATIONS = {augmentation_name: augmentation for augmentation_name, augmentation in getmembers(importlib.import_module('audio_classifier.augmentations.audio'), isclass)}
AVAILABLE_SPECTROGRAM_AUGMENTATIONS = {augmentation_name: augmentation for augmentation_name, augmentation in getmembers(importlib.import_module('audio_classifier.augmentations.spectrogram'), isclass)}
DEFAULT_MODEL = "custom.MNIST_convnet"
