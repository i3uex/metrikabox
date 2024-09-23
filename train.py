import os
import json
import time
import pickle
import argparse
import importlib
from collections import Counter
from inspect import getmembers, isfunction, isclass

from matplotlib import pyplot as plt
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from keras import callbacks
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, CategoricalAccuracy, BinaryAccuracy

from augmentations.spectrogram import SpecAugmentLayer
from loaders import FolderLoader, ClassLoaderFromSameFileName
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, CHECKPOINTS_FOLDER, MODEL_CONFIG_FOLDER
from model.builder import AudioModelBuilder, DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS, DEFAULT_MEL_F_MIN
from augmentations.audio import WhiteNoiseAugmentation

AVAILABLE_KERAS_MODELS = {model_name: model for model_name, model in getmembers(importlib.import_module('tensorflow.keras.applications'), isfunction)}
AVAILABLE_KERAS_OPTIMIZERS = {optimizer_name: optimizer for optimizer_name, optimizer in getmembers(importlib.import_module('tensorflow.keras.optimizers'), isclass)}
AVAILABLE_CLASS_LOADERS = {class_loader_name: class_loader for class_loader_name, class_loader in getmembers(importlib.import_module('loaders.class_loader'), isclass)}
parser = argparse.ArgumentParser(prog='AudioTrain', description='Trains')
parser.add_argument('folder')
parser.add_argument('--model_id', default=None)
parser.add_argument('-sr', '--sample_rate', default=DEFAULT_SAMPLE_RATE, type=int, help='Sample rate the audio will be converted to')
parser.add_argument('--window', default=DEFAULT_WINDOW, type=float, help='Time in seconds to be processed as a single item')
parser.add_argument('--step', default=DEFAULT_STEP, type=float, help='Time in seconds to skip between windows. It is recommendable for step to be maximum half of the window to have overlap between windows and don\'t lose information')
parser.add_argument('--use_mmap', action='store_true', help="Whether to create an intermediate mmap to dump the data to for processing. Useful when no enough RAM is available to fit the audios")
parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int)
parser.add_argument('--stft_nfft', default=DEFAULT_STFT_N_FFT, type=int, help='Length of the FFT window')
parser.add_argument('--stft_win', default=DEFAULT_STFT_WIN, type=int, help='Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft')
parser.add_argument('--stft_hop', default=DEFAULT_STFT_HOP, type=int, help='Number of samples between successive frames')
parser.add_argument('--stft_nmels', default=DEFAULT_N_MELS, type=int, help='Number of Mel bands to generate')
parser.add_argument('--mel_f_min', default=DEFAULT_MEL_F_MIN, type=int, help='Lowest frequency of the mel filterbank')
parser.add_argument('--model', default=None, choices=AVAILABLE_KERAS_MODELS.keys(), help='Any of the models of keras.applications that will be used as classification model')
parser.add_argument('--optimizer', default=None, choices=AVAILABLE_KERAS_OPTIMIZERS.keys(), help='Any of the optimizers of keras.optimizers that will be used as optimizer in model training')
parser.add_argument('--class_loader', default=None, choices=AVAILABLE_CLASS_LOADERS.keys(), help='Any of the available class loaders')
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--trainset_shuffle_size', default=1024, type=int)
args = parser.parse_args()

if not args.folder.endswith("/"):
    args.folder += "/"

if not os.path.exists(MODEL_CONFIG_FOLDER):
    os.mkdir(MODEL_CONFIG_FOLDER)

MODEL_ID = args.model_id
if not MODEL_ID:
    MODEL_ID = f"{int(time.time())}_{args.sample_rate}Hz_{args.window}w_{args.step}s"
MODEL_ID_CONFIG_FOLDER = f'{MODEL_CONFIG_FOLDER}/{MODEL_ID}'
if not os.path.exists(MODEL_ID_CONFIG_FOLDER):
    os.mkdir(MODEL_ID_CONFIG_FOLDER)


model_config = {
    "sample_rate": args.sample_rate,
    "window": args.window,
    "step": args.step,
    "stft_nfft": args.stft_nfft,
    "stft_window": args.stft_win,
    "stft_hop": args.stft_hop,
    "stft_nmels": args.stft_nmels,
    "mel_f_min": args.mel_f_min
}

with open(f'{MODEL_ID_CONFIG_FOLDER}/model-config.json', "w") as f:
    json.dump(model_config, f)


def load_data():
    """
    Loads the data to train the model
    :return: loaded data in a tuple (x, y, num_classes)
    """
    class_loader = ClassLoaderFromSameFileName()
    if args.class_loader:
        class_loader = AVAILABLE_CLASS_LOADERS[args.class_loader]()
    data_loader = FolderLoader(
        sample_rate=args.sample_rate,
        window=args.window,
        step=args.step,
        class_loader=class_loader,
        out_folder=args.folder,
        use_mmap=args.use_mmap
    )
    x, y = data_loader.load(args.folder, classes2avoid=["commercial"])
    assert len(y) == len(x)
    print(Counter(y))
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    with open(f'{MODEL_ID_CONFIG_FOLDER}/LabelEncoder.pkl', "wb") as f:
        pickle.dump(encoder, f)
    return x, y, num_classes


def plot_history(history):
    """
    Plots the history of the model training
    :param history: History object from keras
    :return:
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"{str(MODEL_ID)}_acc.png")
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"{str(MODEL_ID)}_loss.png")
    plt.show()


def train(x, y, num_classes):
    """
    Trains the model
    :param x: Audio data
    :param y: Labels
    :param num_classes: Number of classes
    :return:
    """
    predefined_model = None
    if args.model:
        predefined_model = AVAILABLE_KERAS_MODELS[args.model](
            include_top=False,
            pooling='avg',
            weights=None
        )
    model = AudioModelBuilder(**model_config).get_model(
        num_classes,
        audio_augmentations=[WhiteNoiseAugmentation()],
        spectrum_augmentations=[SpecAugmentLayer(5, 10)],
        predefined_model=predefined_model,
    )
    optimizer = Adam
    if args.optimizer:
        optimizer = AVAILABLE_KERAS_OPTIMIZERS[args.optimizer]

    model.compile(
        optimizer(learning_rate=args.learning_rate),
        loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
        weighted_metrics=['accuracy', Precision(), Recall()]
    )
    val_size = 0.2
    train_num_items = len(x) - round(len(x)*val_size)
    
    print("Preparing datasets")

    output_signature = (
        tf.TensorSpec(shape=(int(args.sample_rate*args.window)), dtype=tf.int16),
        tf.TensorSpec(shape=(num_classes if num_classes > 2 else 1), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.float64)
    )

    def gen(_X, _Y):
        for _x, _y, computed_sample_weight in zip(_X, _Y, class_weight.compute_sample_weight('balanced', _Y)):
            yield _x, _y, computed_sample_weight

    # Prepare the train dataset.
    train_dataset = tf.data.Dataset.from_generator(
        lambda: gen(x[:train_num_items], y[:train_num_items]),
        output_signature=output_signature,
    ).shuffle(train_num_items//10).batch(args.batch_size)

    # Prepare the validation dataset.
    val_dataset = tf.data.Dataset.from_generator(
        lambda: gen(x[train_num_items:], y[train_num_items:]),
        output_signature=output_signature,
    ).batch(args.batch_size)

    checkpoint_filepath = f'{CHECKPOINTS_FOLDER}/{MODEL_ID}/'
    print("Starting training")
    
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=args.epochs,
                        callbacks=[
                            callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True
                            ),
                            callbacks.EarlyStopping(
                                monitor='val_accuracy',
                                min_delta=0.025,
                                verbose=1,
                                patience=50
                            ),
                            callbacks.ReduceLROnPlateau(
                                verbose=1,
                                patience=25
                            ),
                            callbacks.TensorBoard(
                                log_dir=f'logs/{MODEL_ID}/'
                            )
                        ]
    )
    plot_history(history)
    model.save("weights.h5")


if __name__ == '__main__':
    train(*load_data())
