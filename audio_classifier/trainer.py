import json
import os
from typing import List, Tuple, Collection
import tensorflow as tf
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelBinarizer
from audio_classifier import Dataset
from audio_classifier import constants
from audio_classifier.utils import LOGGER


def _generate(x: Collection, y: Collection, class_weight='balanced') -> callable:
    def _gen():
        for _item in zip(x, y, compute_sample_weight(class_weight, y)):
            yield _item

    return _gen


def _dump_model_config(model_id: str, model_config: dict, checkpoints_folder: str = constants.CHECKPOINTS_FOLDER):
    model_id_config_folder = f'{checkpoints_folder}/{constants.MODEL_CONFIG_FOLDER}/{model_id}'
    os.makedirs(model_id_config_folder, exist_ok=True)
    model_config_file_path = f'{model_id_config_folder}/model-config.json'
    with open(model_config_file_path, "w") as f:
        json.dump(model_config, f)
    return model_config_file_path


class Trainer:
    def __init__(
            self,
            predefined_model=None,
            audio_augmentations: List[str] = (),
            spectrogram_augmentations: List[str] = ()
    ):
        self.predefined_model = None
        if predefined_model:
            self.predefined_model = constants.AVAILABLE_MODELS[predefined_model](
                include_top=False,
                pooling='avg',
                weights=None
            )
        for audio_augmentation in audio_augmentations:
            if audio_augmentation not in constants.AVAILABLE_AUDIO_AUGMENTATIONS:
                raise ValueError(f"Audio augmentation {audio_augmentation} not available")
        self.audio_augmentations = [
            constants.AVAILABLE_AUDIO_AUGMENTATIONS[audio_augmentation]()
            for audio_augmentation in audio_augmentations
        ]
        for spectrogram_augmentation in spectrogram_augmentations:
            if spectrogram_augmentation not in constants.AVAILABLE_SPECTROGRAM_AUGMENTATIONS:
                raise ValueError(f"Spectrogram augmentation {spectrogram_augmentation} not available")

        self.spectrogram_augmentations = [
            constants.AVAILABLE_SPECTROGRAM_AUGMENTATIONS[spectrogram_augmentation]()
            for spectrogram_augmentation in spectrogram_augmentations
        ]

    def train(
            self,
            dataset: Dataset,
            val_size: float = 0.2,
            optimizer: str = None,
            learning_rate: float = constants.DEFAULT_LR,
            batch_size: int = constants.DEFAULT_BATCH_SIZE,
            epochs: int = constants.DEFAULT_EPOCHS,
            checkpoints_folder: str = constants.CHECKPOINTS_FOLDER,
            model_id: str = constants.DEFAULT_MODEL_ID,
            early_stopping_patience: int = constants.DEFAULT_EARLY_STOPPING_PATIENCE,
            reduce_lr_on_plateau_patience: int = constants.DEFAULT_REDUCE_LR_ON_PLATEAU_PATIENCE,
            checkpoint_metric=constants.DEFAULT_CHECKPOINT_METRIC,
            early_stopping_metric=constants.DEFAULT_EARLY_STOPPING_METRIC,
    ) -> Tuple[str, str, tf.keras.callbacks.History]:

        x, y = dataset.load()

        # Create Label Encoder and dump model config
        encoder = LabelBinarizer()
        y = encoder.fit_transform(y)
        num_classes = len(encoder.classes_)
        model_config = dataset.get_config()
        model_config["classes"] = encoder.classes_.tolist()
        model_config_path = _dump_model_config(model_id, model_config)
        model_builder = dataset.get_model_builder(model_config)
        model = model_builder.get_model(
            num_classes,
            audio_augmentations=self.audio_augmentations,
            spectrum_augmentations=self.spectrogram_augmentations,
            predefined_model=self.predefined_model,
        )

        # Prepare model optimizer
        if optimizer:
            optimizer = constants.AVAILABLE_KERAS_OPTIMIZERS[optimizer]
        else:
            optimizer = tf.keras.optimizers.Adam
        model.compile(
            optimizer(learning_rate=learning_rate),
            loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
            weighted_metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy") if num_classes > 2 else tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall")
            ]
        )

        # Prepare model output signature
        output_signature = (
            model_builder.get_output_signature(),
            tf.TensorSpec(shape=(num_classes if num_classes > 2 else 1), dtype=tf.int64),
            tf.TensorSpec(shape=(), dtype=tf.float64)
        )

        # Prepare datasets
        train_num_items = len(x) - round(len(x) * val_size)

        train_dataset = tf.data.Dataset.from_generator(
            _generate(x[:train_num_items], y[:train_num_items]),
            output_signature=output_signature,
        ).shuffle(train_num_items).batch(batch_size, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_generator(
            _generate(x[train_num_items:], y[train_num_items:]),
            output_signature=output_signature,
        ).batch(batch_size, drop_remainder=True)

        checkpoint_filepath = f'{checkpoints_folder}/{model_id}.keras'

        # Perform training
        LOGGER.info("Starting training")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                monitor=checkpoint_metric,
                save_best_only=True
            ),
              tf.keras.callbacks.TensorBoard(
                  log_dir=f'logs/{model_id}/'
              )
        ]
        if early_stopping_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    verbose=1,
                    min_delta=0.025,
                    monitor=early_stopping_metric,
                    patience=early_stopping_patience
                )
            )
        if True:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    verbose=1,
                    patience=reduce_lr_on_plateau_patience
                )
            )
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        os.makedirs("histories", exist_ok=True)
        with open(f'histories/{model_id}_history.json', "w") as f:
            json.dump(history.history, f, default=str)
        return checkpoint_filepath, model_config_path, history
