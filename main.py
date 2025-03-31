import datetime
import json
import os
from typing import List, Union
import fire
from matplotlib import pyplot as plt
from audio_classifier import Trainer
from audio_classifier.dataset import TYPE2DATASET
from audio_classifier.infer import TASK2MODEL
from audio_classifier import constants
from audio_classifier.loaders.data_loaders import TYPE2LOADER
from audio_classifier.utils import LOGGER


class Main:

    def infer(
            self,
            filename,
            model_path,
            model_config_path=None,
            loader_type='Audio',
            task='segment'
    ):
        if __name__ == '__main__':
            if task not in TASK2MODEL:
                LOGGER.error(f"Task {task} is not supported. Supported tasks are {TASK2MODEL.keys()}")
                exit()
            if not model_config_path:
                base_path, model_name = model_path.rsplit('.', 1)[0].rsplit('/', 1)
                model_config_path = f"{base_path}/{constants.MODEL_CONFIG_FOLDER}/{model_name}/model-config.json"
            model = TASK2MODEL[task](model_path, model_config_path, TYPE2LOADER[loader_type])
            base_file_name = filename.split(".", 1)[0]
            probabilities = model.predict_without_format(filename)
            with open(f"{base_file_name}_probas.json", 'w') as f:
                json.dump(probabilities.tolist(), f)
            predictions = model.format_output(probabilities)
            with open(f"{base_file_name}_{task}.json", 'w') as f:
                json.dump(predictions, f, default=str)
            print(predictions)

    def train(
            self,
            folder,
            sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
            window: float = constants.DEFAULT_WINDOW,
            step: float = constants.DEFAULT_STEP,
            classes2avoid: Union[List[str], str] = (),
            checkpoints_folder: str = constants.CHECKPOINTS_FOLDER,
            optimizer: str = constants.DEFAULT_OPTIMIZER,
            class_loader: str = constants.DEFAULT_CLASS_LOADER,
            learning_rate: float = constants.DEFAULT_LR,
            model_id: str = constants.DEFAULT_MODEL_ID,
            stft_nfft: int = constants.DEFAULT_STFT_N_FFT,
            stft_win: int = constants.DEFAULT_STFT_WIN,
            stft_hop: int = constants.DEFAULT_STFT_HOP,
            stft_nmels: int = constants.DEFAULT_N_MELS,
            mel_f_min: int = constants.DEFAULT_MEL_F_MIN,
            model: str = constants.DEFAULT_MODEL,
            audio_augmentations: List[str] = (),
            spectrogram_augmentations: List[str] = (),
            epochs=constants.DEFAULT_EPOCHS,
            batch_size=constants.DEFAULT_BATCH_SIZE,
            dataset_type=constants.DEFAULT_DATASET_TYPE,
            encodec_model=constants.DEFAULT_ENCODEC_MODEL,
            encodec_decode=constants.DEFAULT_ENCODEC_DECODE,
            bandwidth=constants.DEFAULT_ENCODEC_BANDWIDTH,
            early_stopping_patience=constants.DEFAULT_EARLY_STOPPING_PATIENCE,
            early_stopping_metric=constants.DEFAULT_EARLY_STOPPING_METRIC,
            reduce_lr_on_plateau_patience=constants.DEFAULT_REDUCE_LR_ON_PLATEAU_PATIENCE,
            checkpoint_metric=constants.DEFAULT_CHECKPOINT_METRIC
    ):
        """
        Trains the model
        :param checkpoint_metric: Metric used to obtain the best checkpoint of the model
        :param reduce_lr_on_plateau_patience: Patience for reducing the learning rate on plateau (0 for no reducing)
        :param early_stopping_metric: Metric used to monitor the early stopping
        :param early_stopping_patience: Patience for early stopping (0 for no early stopping)
        :param bandwidth: Bandwidth the audio was encoded to (Only use with 'EnCodec' dataset type)
        :param encodec_decode: Whether if the audio should be decoded. Increases latent space (Only use with 'EnCodec' dataset type)
        :param encodec_model: EnCodec Model the audios where encoded with (Only use with 'EnCodec' dataset type)
        :param dataset_type: Format of files in the dataset
        :param folder: Path to the older containing the audio files
        :param sample_rate: Sample rate the audios will be resampled to
        :param window: Seconds of audio to use
        :param step: Seconds to move the window
        :param classes2avoid: Classes to avoid from training model
        :param checkpoints_folder: Path to save the model checkpoints
        :param optimizer: String with the optimizer to use
        :param class_loader: String with the class loader to use
        :param learning_rate: Learning rate for the optimizer
        :param model_id: String with the model id
        :param stft_nfft: Number of FFTs to use
        :param stft_win: Length of the STFT window
        :param stft_hop: Length of the STFT hop
        :param stft_nmels: Number of mel bands to use
        :param mel_f_min: Minimum frequency for the mel bands
        :param model: Name of the predefined model to use. Any of keras.applications
        :param audio_augmentations: List of audio augmentations to use
        :param spectrogram_augmentations: List of spectrogram augmentations to use
        :param batch_size: Batch size for the training
        :param epochs: Number of epochs to train
        :return:
        """
        if type(classes2avoid) is str:
            classes2avoid = classes2avoid.split(",")
        trainer = Trainer(
            predefined_model=model,
            audio_augmentations=audio_augmentations,
            spectrogram_augmentations=spectrogram_augmentations
        )
        dataset = TYPE2DATASET[dataset_type](
            folder=folder,
            sample_rate=sample_rate,
            window=window,
            step=step,
            stft_nfft=stft_nfft,
            stft_win=stft_win,
            stft_hop=stft_hop,
            stft_nmels=stft_nmels,
            mel_f_min=mel_f_min,
            classes2avoid=classes2avoid,
            class_loader=class_loader,
            model=encodec_model,
            decode=encodec_decode,
            bandwidth=bandwidth
        )
        if not model_id:
            folder_name = folder.rsplit("/")[-1] if folder[-1] != "/" else folder.rsplit("/")[-2]
            model_id = f"{folder_name}_{str(datetime.datetime.now())}_{sample_rate}Hz_{window}w_{step}s"
        checkpoint_filepath, model_config_path, history = trainer.train(
            dataset,
            val_size=0.2,
            optimizer=optimizer,
            learning_rate=learning_rate,
            checkpoints_folder=checkpoints_folder,
            batch_size=batch_size,
            epochs=epochs,
            model_id=model_id,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
            checkpoint_metric=checkpoint_metric,
            early_stopping_metric=early_stopping_metric,
        )
        os.makedirs('histories', exist_ok=True)
        with open(f'histories/{model_id}.json', "w") as f:
            json.dump(history.history, f, default=str)
        return checkpoint_filepath, model_config_path, history


if __name__ == '__main__':
    fire.Fire(Main)