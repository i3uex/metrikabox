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
from tensorflow.data import Dataset
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

from loaders import FolderLoader, ClassLoaderFromSameFileName
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, CHECKPOINTS_FOLDER, MODEL_CONFIG_FOLDER
from model.model import AudioModelBuilder, DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS, DEFAULT_MEL_F_MIN
from augmentations.audio import WhiteNoiseAugmentation

AVAILABLE_KERAS_MODELS = {model_name: model for model_name, model in getmembers(importlib.import_module('tensorflow.keras.applications'), isfunction)}
AVAILABLE_KERAS_OPTIMIZERS = {optimizer_name: optimizer for optimizer_name, optimizer in getmembers(importlib.import_module('tensorflow.keras.optimizers'), isclass)}
AVAILABLE_CLASS_LOADERS = {class_loader_name: class_loader for class_loader_name, class_loader in getmembers(importlib.import_module('loaders.class_loader'), isclass)}
parser = argparse.ArgumentParser(prog = 'AudioTrain', description = 'Trains')
parser.add_argument('folder')
parser.add_argument('--model_id', default=int(time.time()))
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
    assert len(y) == x.shape[0]
    print(Counter(y))
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    with open(f'{MODEL_ID_CONFIG_FOLDER}/LabelEncoder.pkl', "wb") as f:
        pickle.dump(encoder, f)
    return x, y, num_classes

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.savefig(str(MODEL_ID) + ".png")

def train(x, y, num_classes):
    predefined_model = None
    if args.model:
        predefined_model = AVAILABLE_KERAS_MODELS[args.model](
            include_top=False, 
            pooling='avg', 
            weights=None
        )
    model = AudioModelBuilder(**model_config).get_model(num_classes,
                                                        predefined_model=predefined_model,
                                                        )
    optimizer = Adam
    if args.optimizer:
        optimizer = AVAILABLE_KERAS_OPTIMIZERS[args.optimizer]

    model.compile(
        optimizer(learning_rate=args.learning_rate),
        loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
        metrics=['accuracy', Precision(), Recall()]
    )
    val_size = 0.2
    num_items = round(len(x)*val_size)
    print("Preparing datasets")
    # Prepare the validation dataset.
    val_dataset = Dataset.from_tensor_slices((x[-num_items:], y[-num_items:], class_weight.compute_sample_weight('balanced', y[-num_items:]))).batch(args.batch_size)

    x, y = x[:-num_items], y[:-num_items]

    train_dataset = Dataset.from_tensor_slices((x, y, class_weight.compute_sample_weight('balanced', y)))
    train_dataset = train_dataset.shuffle(args.trainset_shuffle_size).batch(args.batch_size)
    
    del x, y
    
    checkpoint_filepath = f'{CHECKPOINTS_FOLDER}/{MODEL_ID}/'
    print("Starting training")
   
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=args.epochs,
                        callbacks=[
                            callbacks.ModelCheckpoint(
                                filepath=checkpoint_filepath,
                                monitor='val_loss',
                                mode='max',
                                save_best_only=True
                            ),
                            callbacks.EarlyStopping(
                                monitor='val_accuracy',
                                min_delta=0.0025,
                                verbose=1,
                                patience=25
                            ),
                            callbacks.ReduceLROnPlateau(
                                verbose=1
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
