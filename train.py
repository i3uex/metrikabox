import argparse
import sys
import time
from collections import Counter

import pickle
from tensorflow.keras import callbacks
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from kapre.augmentation import SpecAugment

from config import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP, USE_MMAP, DEFAULT_BATCH_SIZE
from loaders import FolderLoader, ClassLoaderFromFolderName
from model.model import AudioModelBuilder

parser = argparse.ArgumentParser(prog = 'AudioTrain', description = 'Trains')
parser.add_argument('folder')
parser.add_argument('--model_id', default=int(time.time()))
parser.add_argument('-sr', '--sample_rate', default=SAMPLE_RATE, type=int)
parser.add_argument('--window', default=CONTEXT_WINDOW, type=float)
parser.add_argument('--step', default=PROCESSING_STEP, type=float)
parser.add_argument('--use_mmap', default=USE_MMAP, type=bool)
parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
args = parser.parse_args()

data_loader = FolderLoader(sample_rate=args.sample_rate, window=args.window, step=args.step, class_loader=ClassLoaderFromFolderName(), out_folder=args.folder, use_mmap=False)
X, Y = data_loader.load(sys.argv[1])

assert len(Y) == X.shape[0]

MODEL_ID = args.model_id
print(Counter(Y))
print(len(Counter(Y)))
encoder = LabelBinarizer()
Y = encoder.fit_transform(Y)
num_classes = len(encoder.classes_)
with open("LabelEncoder-%d.pkl" % MODEL_ID, "wb") as f:
    pickle.dump(encoder, f)

# Add the spec_augment layer for augmentations
spec_augment = SpecAugment(
    freq_mask_param=5,
    time_mask_param=10,
    n_freq_masks=5,
    n_time_masks=3,
    data_format='channels_last'
)

model = AudioModelBuilder(sample_rate=args.sample_rate, context_window=args.window, processing_step=args.step).get_model(num_classes)

model.compile("adam", loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy", metrics=['accuracy'])

print("Computing sample weight")
checkpoint_filepath = 'checkpoints/%d/' % MODEL_ID
model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)
tboard = callbacks.TensorBoard()
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, min_delta=0.02, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.02, patience=15, mode='auto')

print("Starting training")
model.fit(X, Y, validation_split=0.2, epochs=100, shuffle=True, batch_size=128, sample_weight=class_weight.compute_sample_weight('balanced', Y), callbacks=[model_checkpoint_callback, reduce_lr, tboard])
