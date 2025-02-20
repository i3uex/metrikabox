import datetime
import json
import os
from typing import List
import fire
from matplotlib import pyplot as plt
from audio_classifier import Dataset
from audio_classifier import AudioClassifier, AudioSegmenter, Trainer
from audio_classifier.config import DEFAULT_SAMPLE_RATE, DEFAULT_STEP, DEFAULT_WINDOW, CHECKPOINTS_FOLDER
from audio_classifier.model.builder import DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS, DEFAULT_MEL_F_MIN

TASK2MODEL = {
    'classify': AudioClassifier,
    'segment': AudioSegmenter,
}


def plot_history(history, model_id):
    """
    Plots the history of the model training
    :param history: History object from keras
    :return:
    """
    fig, (ax, bx) = plt.subplots(2, 1)
    ax.plot(history.history['binary_accuracy'] if 'binary_accuracy' in history.history else history.history['categorical_accuracy'])
    ax.plot(history.history['val_binary_accuracy'] if 'val_binary_accuracy' in history.history else history.history['val_categorical_accuracy'])
    ax.set_title('model accuracy')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    ax.legend(['train', 'val'], loc='upper left')
    bx.plot(history.history['loss'])
    bx.plot(history.history['val_loss'])
    bx.set_title('model loss')
    bx.set_ylabel('loss')
    bx.set_xlabel('epoch')
    bx.legend(['train', 'val'], loc='upper left')
    plt.savefig(model_id + '.png')
    plt.show()


class Main:

    def predict(
            self,
            filename,
            model_id,
            task='segment'
    ):
        if __name__ == '__main__':
            if task not in TASK2MODEL:
                print(f"Task {task} is not supported. Supported tasks are {TASK2MODEL.keys()}")
                exit()
            model = TASK2MODEL[task](model_id)
            base_file_name = filename.split(".", 1)[0]
            probabilities = model.predict_without_format(filename)
            with open(f"{base_file_name}_probas.json", 'w') as f:
                json.dump(probabilities, f)
            predictions = model.format_output(probabilities)
            with open(f"{base_file_name}.json", 'w') as f:
                json.dump(predictions, f)

    def train(
            self,
            folder,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            window: float = DEFAULT_WINDOW,
            step: float = DEFAULT_STEP,
            classes2avoid: List[str] = (),
            checkpoints_folder: str = CHECKPOINTS_FOLDER,
            optimizer: str = "Adam",
            class_loader: str = "ClassLoaderFromFolderName",
            learning_rate: float = 0.001,
            model_id: str = None,
            stft_nfft: int = DEFAULT_STFT_N_FFT,
            stft_win: int = DEFAULT_STFT_WIN,
            stft_hop: int = DEFAULT_STFT_HOP,
            stft_nmels: int = DEFAULT_N_MELS,
            mel_f_min: int = DEFAULT_MEL_F_MIN,
            model: str = None,
            audio_augmentations: List[str] = (),
            spectrogram_augmentations: List[str] = ()
    ):
        """
        Trains the model
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
        :return:
        """
        trainer = Trainer(
            stft_nfft=stft_nfft,
            stft_win=stft_win,
            stft_hop=stft_hop,
            stft_nmels=stft_nmels,
            mel_f_min=mel_f_min,
            predefined_model=model,
            audio_augmentations=audio_augmentations,
            spectrogram_augmentations=spectrogram_augmentations
        )
        dataset = Dataset(
            folder,
            sample_rate=sample_rate,
            window=window,
            step=step,
            classes2avoid=classes2avoid,
            class_loader=class_loader
        )
        if not model_id:
            folder_name = folder.rsplit("/")[-1] if folder[-1] != "/" else folder.rsplit("/")[-2]
            model_id = f"{folder_name}_{str(datetime.datetime.now())}_{sample_rate}Hz_{window}w_{step}s"
        model, history = trainer.train(
            dataset,
            val_size=0.2,
            optimizer=optimizer,
            learning_rate=learning_rate,
            checkpoints_folder=checkpoints_folder,
            model_id=model_id,
        )
        os.makedirs('histories', exist_ok=True)
        with open(f'histories/{model_id}.json', "w") as f:
            json.dump(history.history, f, default=str)
        plot_history(history, model_id)


if __name__ == '__main__':
    fire.Fire(Main)