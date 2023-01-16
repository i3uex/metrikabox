import argparse
import importlib
import json
import sys
import time
from collections import Counter
import pickle
from inspect import getmembers, isfunction

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.utils import class_weight
from config import SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP, USE_MMAP, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from model.model import AudioModelBuilder, DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS
from loaders import FolderLoader, ClassLoaderFromFolderName
from matplotlib import pyplot as plt

AVAILABLE_KERAS_MODELS = {model_name: model for model_name, model in getmembers(importlib.import_module('tensorflow.keras.applications'), isfunction)}
AVAILABLE_KERAS_OPTIMIZERS = {optimizer_name: optimizer for optimizer_name, optimizer in getmembers(importlib.import_module('tensorflow.keras.optimizers'), isfunction)}
parser = argparse.ArgumentParser(prog = 'AudioTrain', description = 'Trains')
parser.add_argument('folder')
parser.add_argument('--model_id', default=int(time.time()))
parser.add_argument('-sr', '--sample_rate', default=SAMPLE_RATE, type=int, help='Sample rate the audio will be converted to')
parser.add_argument('--window', default=DEFAULT_WINDOW, type=float, help='Time in seconds to be processed as a single item')
parser.add_argument('--step', default=DEFAULT_STEP, type=float, help='Time in seconds to skip between windows. It is recommendable for step to be maximum half of the window to have overlap between windows and don\'t lose information')
parser.add_argument('--use_mmap', default=USE_MMAP, type=bool)
parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int)
parser.add_argument('--stft_nfft', default=DEFAULT_STFT_N_FFT, type=int, help='Length of the FFT window')
parser.add_argument('--stft_win', default=DEFAULT_STFT_WIN, type=int, help='Each frame of audio is windowed by window of length win_length and then padded with zeros to match n_fft')
parser.add_argument('--stft_hop', default=DEFAULT_STFT_HOP, type=int, help='Number of samples between successive frames')
parser.add_argument('--stft_nmels', default=DEFAULT_N_MELS, type=int, help='Number of Mel bands to generate')
parser.add_argument('--model', default=None, choices=AVAILABLE_KERAS_MODELS.keys(), help='Any of the models of keras.applications that will be used as classification model')
parser.add_argument('--optimizer', default=None, choices=AVAILABLE_KERAS_OPTIMIZERS.keys(), help='Any of the optimizers of keras.optimizers that will be used as optimizer in model training')
parser.add_argument('--learning_rate', default=0.001, type=float)
args = parser.parse_args()

MODEL_ID = args.model_id

model_config = {
    "sample_rate": args.sample_rate,
    "window": args.window,
    "step": args.step,
    "stft_nfft": args.stft_nfft,
    "stft_window": args.stft_win,
    "stft_hop": args.stft_hop,
    "stft_nmels": args.stft_nmels
}

with open("model-config-%s.json" % MODEL_ID, "w") as f:
    json.dump(model_config, f)

def load_data():
    data_loader = FolderLoader(sample_rate=args.sample_rate, window=args.window, step=args.step, class_loader=ClassLoaderFromFolderName(), out_folder=args.folder, use_mmap=False)
    x, y = data_loader.load(sys.argv[1])
    assert len(y) == x.shape[0]
    print(Counter(y))
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    with open("LabelEncoder-%s.pkl" % MODEL_ID, "wb") as f:
        pickle.dump(encoder, f)
    return x, y, num_classes

def train(x, y, num_classes):
    predefined_model = None
    if args.model:
        predefined_model = AVAILABLE_KERAS_MODELS[args.model]
    model = AudioModelBuilder(**model_config).get_model(num_classes, predefined_model=predefined_model)
    optimizer = Adam
    if args.optimizer:
        optimizer = AVAILABLE_KERAS_OPTIMIZERS[args.optimizer]
    model.compile(optimizer(learning_rate=args.learning_rate), loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy", metrics=['accuracy'])
    checkpoint_filepath = 'checkpoints/%s/' % MODEL_ID
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    tboard = callbacks.TensorBoard()
    reduce_lr = callbacks.ReduceLROnPlateau(verbose=1, min_delta=0.1, min_lr=1e-6)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, mode='auto', verbose=1)
    print("Starting training")
    history = model.fit(x, y, validation_split=0.2, epochs=args.epochs, shuffle=True, batch_size=args.batch_size,
              sample_weight=class_weight.compute_sample_weight('balanced', y),
              callbacks=[model_checkpoint_callback, reduce_lr, tboard, early_stopping])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig(MODEL_ID + ".png")

if __name__ == '__main__':
    train(*load_data())