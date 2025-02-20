import os
import json
import argparse
from matplotlib import pyplot as plt
from classes import Trainer
from classes.dataset import Dataset
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, CHECKPOINTS_FOLDER
from model.builder import DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS, DEFAULT_MEL_F_MIN
from constants import *

parser = argparse.ArgumentParser(prog='AudioTrain', description='Trains')
parser.add_argument('folder', help='Folder with the dataset to be used')
parser.add_argument('--model_id', default=None, help='Model id to be used')
parser.add_argument('-sr', '--sample_rate', default=DEFAULT_SAMPLE_RATE, type=int, help='Sample rate the audio will be converted to')
parser.add_argument('--window', default=DEFAULT_WINDOW, type=float, help='Time in seconds to be processed as a single item')
parser.add_argument('--step', default=DEFAULT_STEP, type=float, help='Time in seconds to skip between windows. It is recommendable for step to be maximum half of the window to have overlap between windows and don\'t lose information')
parser.add_argument('--use_mmap', action='store_true', help="Whether to create an intermediate mmap to dump the data to for processing. Useful when no enough RAM is available to fit the audios")
parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int, help='Batch size for the training')
parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int, help='Number of epochs to train the model')
parser.add_argument('--stft_nfft', default=DEFAULT_STFT_N_FFT, type=int, help='Length of the FFT window')
parser.add_argument('--stft_win', default=DEFAULT_STFT_WIN, type=int, help='Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft')
parser.add_argument('--stft_hop', default=DEFAULT_STFT_HOP, type=int, help='Number of samples between successive frames')
parser.add_argument('--stft_nmels', default=DEFAULT_N_MELS, type=int, help='Number of Mel bands to generate')
parser.add_argument('--mel_f_min', default=DEFAULT_MEL_F_MIN, type=int, help='Lowest frequency of the mel filterbank')
parser.add_argument('--model', default=None, choices=AVAILABLE_KERAS_MODELS.keys(), help='Any of the models of keras.applications that will be used as classification model')
parser.add_argument('--optimizer', default=None, choices=AVAILABLE_KERAS_OPTIMIZERS.keys(), help='Any of the optimizers of keras.optimizers that will be used as optimizer in model training')
parser.add_argument('--class_loader', default=None, choices=AVAILABLE_CLASS_LOADERS.keys(), help='Any of the available class loaders')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate for the optimizer')
parser.add_argument('--trainset_shuffle_size', default=1024, type=int, help='Size of the shuffle buffer for the train dataset')
parser.add_argument('--audio_augmentations', default=[], nargs='+', choices=AVAILABLE_AUDIO_AUGMENTATIONS.keys(), help='Any of the available audio augmentations. No one will be used if not specified')
parser.add_argument('--spectrogram_augmentations', default=[], nargs='+', choices=AVAILABLE_SPECTROGRAM_AUGMENTATIONS.keys(), help='Any of the available spectrogram augmentations. No one will be used if not specified')
parser.add_argument('--classes2avoid', default=[], nargs='+', help='Classes to avoid in the dataset (not to be loaded)')

args = parser.parse_args()

if not args.folder.endswith("/"):
    args.folder += "/"

MODEL_ID = args.model_id


def plot_history(history):
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
    plt.show()


def train():
    """
    Trains the model
    :param x: Audio data
    :param y: Labels
    :param num_classes: Number of classes
    :return:
    """
    trainer = Trainer(
        stft_nfft=args.stft_nfft,
        stft_win=args.stft_win,
        stft_hop=args.stft_hop,
        stft_nmels=args.stft_nmels,
        mel_f_min=args.mel_f_min,
        predefined_model=args.model,
        audio_augmentations=args.audio_augmentations,
        spectrogram_augmentations=args.spectrogram_augmentations
    )
    dataset = Dataset(
        args.folder,
        sample_rate=args.sample_rate,
        window=args.window,
        step=args.step,
        classes2avoid=args.classes2avoid,
        class_loader=args.class_loader
    )
    model, history = trainer.train(
        dataset,
        val_size=0.2,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        checkpoints_folder=CHECKPOINTS_FOLDER,
        model_id=MODEL_ID,
    )
    plot_history(history)
    os.makedirs('histories', exist_ok=True)
    with open(f'histories/{MODEL_ID}.json', "w") as f:
        json.dump(history.history, f, default=str)


if __name__ == '__main__':
    train()
