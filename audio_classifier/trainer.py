import json
import os
import time
from typing import List, Tuple, Collection
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer

from audio_classifier import Dataset
from audio_classifier.config import MODEL_CONFIG_FOLDER, DEFAULT_BATCH_SIZE, \
    DEFAULT_EPOCHS, CHECKPOINTS_FOLDER
from audio_classifier.model import AudioModelBuilder, DEFAULT_STFT_HOP, DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_N_MELS, \
    DEFAULT_MEL_F_MIN
from audio_classifier.constants import AVAILABLE_KERAS_MODELS, AVAILABLE_AUDIO_AUGMENTATIONS, AVAILABLE_SPECTROGRAM_AUGMENTATIONS, \
    AVAILABLE_KERAS_OPTIMIZERS


def generate(x: Collection, y: Collection) -> callable:
    def _gen():
        for _item in zip(x, y, class_weight.compute_sample_weight('balanced', y)):
            yield _item

    return _gen


class Trainer:
    def __init__(
            self,
            stft_nfft: int = DEFAULT_STFT_N_FFT,
            stft_win: int = DEFAULT_STFT_WIN,
            stft_hop: int = DEFAULT_STFT_HOP,
            stft_nmels: int = DEFAULT_N_MELS,
            mel_f_min: int = DEFAULT_MEL_F_MIN,
            predefined_model=None,
            audio_augmentations: List[str] = (),
            spectrogram_augmentations: List[str] = ()
    ):
        self.model_config = {
            "stft_nfft": stft_nfft,
            "stft_window": stft_win,
            "stft_hop": stft_hop,
            "stft_nmels": stft_nmels,
            "mel_f_min": mel_f_min
        }
        self.predefined_model = None
        if predefined_model:
            self.predefined_model = AVAILABLE_KERAS_MODELS[predefined_model](
                include_top=False,
                pooling='avg',
                weights=None
            )
        for audio_augmentation in audio_augmentations:
            if audio_augmentation not in AVAILABLE_AUDIO_AUGMENTATIONS:
                raise ValueError(f"Audio augmentation {audio_augmentation} not available")
        self.audio_augmentations = [
            AVAILABLE_AUDIO_AUGMENTATIONS[audio_augmentation]()
            for audio_augmentation in audio_augmentations
        ]
        for spectrogram_augmentation in spectrogram_augmentations:
            if spectrogram_augmentation not in AVAILABLE_SPECTROGRAM_AUGMENTATIONS:
                raise ValueError(f"Spectrogram augmentation {spectrogram_augmentation} not available")

        self.spectrogram_augmentations = [
            AVAILABLE_SPECTROGRAM_AUGMENTATIONS[spectrogram_augmentation]()
            for spectrogram_augmentation in spectrogram_augmentations
        ]

    def _get_model(self, num_classes: int, sample_rate: int, window: float, step: float):
        model_config = self.model_config.copy()
        model_config.update({
            "sample_rate": sample_rate,
            "window": window,
            "step": step
        })
        return AudioModelBuilder(**model_config).get_model(
            num_classes,
            audio_augmentations=self.audio_augmentations,
            spectrum_augmentations=self.spectrogram_augmentations,
            predefined_model=self.predefined_model,
        )

    def _dump_model_config(self, model_id: str):
        model_id_config_folder = f'{MODEL_CONFIG_FOLDER}/{model_id}'
        if not os.path.exists(model_id_config_folder):
            os.mkdir(model_id_config_folder)
        with open(f'{model_id_config_folder}/model-config.json', "w") as f:
            json.dump(self.model_config, f)

    def train(
            self,
            dataset: Dataset,
            val_size: float = 0.2,
            optimizer: str = None,
            learning_rate: float = 0.001,
            batch_size: int = DEFAULT_BATCH_SIZE,
            epochs: int = DEFAULT_EPOCHS,
            checkpoints_folder: str = CHECKPOINTS_FOLDER,
            model_id: str = "model"
    ) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:

        if not model_id:
            model_id = f"{int(time.time())}_{dataset.sample_rate}Hz_{dataset.window}w_{dataset.step}s"

        x, y = dataset.load()

        # Create Label Encoder and dump model config
        encoder = LabelBinarizer()
        y = encoder.fit_transform(y)
        num_classes = len(encoder.classes_)
        self._dump_model_config(model_id)
        model = self._get_model(num_classes, dataset.sample_rate, dataset.window, dataset.step)

        # Prepare model optimizer
        if optimizer:
            optimizer = AVAILABLE_KERAS_OPTIMIZERS[optimizer]
        else:
            optimizer = tf.keras.optimizers.Adam
        model.compile(
            optimizer(learning_rate=learning_rate),
            loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
            weighted_metrics=[
                tf.keras.metrics.CategoricalAccuracy() if num_classes > 2 else tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )

        # Prepare model output signature
        output_signature = (
            tf.TensorSpec(shape=(int(dataset.sample_rate * dataset.window)), dtype=tf.int16),
            tf.TensorSpec(shape=(num_classes if num_classes > 2 else 1), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.float64)
        )

        # Prepare datasets
        train_num_items = len(x) - round(len(x) * val_size)

        train_dataset = tf.data.Dataset.from_generator(
            generate(x[:train_num_items], y[:train_num_items]),
            output_signature=output_signature,
        ).shuffle(train_num_items // 10).batch(batch_size, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_generator(
            generate(x[train_num_items:], y[train_num_items:]),
            output_signature=output_signature,
        ).batch(batch_size, drop_remainder=True)

        checkpoint_filepath = f'{checkpoints_folder}/{model_id}.keras'

        # Perform training
        print("Starting training")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[
              tf.keras.callbacks.ModelCheckpoint(
                  filepath=checkpoint_filepath,
                  monitor='val_loss',
                  mode='min',
                  save_best_only=True
              ),
              tf.keras.callbacks.EarlyStopping(
                  monitor='val_accuracy',
                  mode="min",
                  min_delta=0.025,
                  verbose=1,
                  patience=50
              ),
              tf.keras.callbacks.ReduceLROnPlateau(
                  verbose=1,
                  patience=25
              ),
              tf.keras.callbacks.TensorBoard(
                  log_dir=f'logs/{model_id}/'
              )
            ]
        )
        return model, history
