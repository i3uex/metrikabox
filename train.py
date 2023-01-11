import argparse
import json
import sys
import time
from collections import Counter
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import callbacks
from sklearn.utils import class_weight
from config import SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP, USE_MMAP, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from model.model import AudioModelBuilder, DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS
from loaders import FolderLoader, ClassLoaderFromFolderName

parser = argparse.ArgumentParser(prog = 'AudioTrain', description = 'Trains')
parser.add_argument('folder')
parser.add_argument('--model_id', default=int(time.time()))
parser.add_argument('-sr', '--sample_rate', default=SAMPLE_RATE, type=int)
parser.add_argument('--window', default=DEFAULT_WINDOW, type=float)
parser.add_argument('--step', default=DEFAULT_STEP, type=float)
parser.add_argument('--use_mmap', default=USE_MMAP, type=bool)
parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
parser.add_argument('--epochs', default=DEFAULT_EPOCHS, type=int)
parser.add_argument('--stft_nfft', default=DEFAULT_STFT_N_FFT, type=int)
parser.add_argument('--stft_win', default=DEFAULT_STFT_WIN, type=int)
parser.add_argument('--stft_hop', default=DEFAULT_STFT_HOP, type=int)
parser.add_argument('--stft_nmels', default=DEFAULT_N_MELS, type=int)
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
    model = AudioModelBuilder(**model_config).get_model(num_classes)
    model.compile("adam", loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
                  metrics=['accuracy'])
    checkpoint_filepath = 'checkpoints/%s/' % MODEL_ID
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    tboard = callbacks.TensorBoard()
    reduce_lr = callbacks.ReduceLROnPlateau(verbose=1, min_delta=0.1, min_lr=1e-6)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=15, mode='auto', verbose=1)
    print("Starting training")
    model.fit(x, y, validation_split=0.2, epochs=100, shuffle=True, batch_size=128,
              sample_weight=class_weight.compute_sample_weight('balanced', y),
              callbacks=[model_checkpoint_callback, reduce_lr, tboard, early_stopping])


if __name__ == '__main__':
    train(*load_data())