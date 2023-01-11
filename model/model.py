from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from kapre.composed import get_melspectrogram_layer

from config import SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP
from loaders import FileLoader
from model.classification.MNIST import MNIST_convnet
from utils import get_mels_from_hop_and_win_lengths

DEFAULT_STFT_N_FFT = 1024
DEFAULT_STFT_WIN = 1024
DEFAULT_STFT_HOP = 256
DEFAULT_N_MELS = 128


class AudioModelBuilder:
    def __init__(self, sample_rate=SAMPLE_RATE, window=DEFAULT_WINDOW, step=DEFAULT_STEP, stft_nfft=DEFAULT_STFT_N_FFT, stft_window=DEFAULT_STFT_WIN, stft_hop=DEFAULT_STFT_HOP, stft_nmels=DEFAULT_N_MELS):
        self.sample_rate = sample_rate
        self.window = window
        self.step = step
        self.stft_nfft = stft_nfft
        self.stft_window = stft_window
        self.stft_hop = stft_hop
        self.stft_nmels = stft_nmels
        self.file_loader = FileLoader(sample_rate=self.sample_rate, window=self.window, step=self.step)

    def __pretrained_model_with_conv(self, num_classes, pretrained_model=MNIST_convnet):
        input_tensor = layers.Input(shape=(get_mels_from_hop_and_win_lengths(self.stft_hop, self.stft_window, input_size=int(self.sample_rate*self.window)), self.stft_nmels, 1))
        convolution_layer = layers.Conv2D(3, (3, 3), padding='same')(input_tensor)  # X has a dimension of (IMG_SIZE,N_MELS,3)
        base_model = pretrained_model(
            include_top=False,
            #weights='imagenet',
            weights=None,
            pooling='avg'
        )
        for layer in base_model.layers:
            layer.trainable = True  # trainable has to be false in order to freeze the layers
        base_model = base_model(convolution_layer)
        op = layers.Dense(256, activation='relu')(base_model)
        op = layers.Dropout(.25)(op)
        output_tensor = layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')(op)
        return Model(inputs=input_tensor, outputs=output_tensor)

    def get_melspectrogram(self):
        return get_melspectrogram_layer(
            n_fft=self.stft_nfft,
            win_length=self.stft_window,
            hop_length=self.stft_hop,
            n_mels=self.stft_nmels,
            sample_rate=self.sample_rate,
            return_decibel=True,
            input_data_format='channels_last', output_data_format='channels_last',
            input_shape=(int(self.sample_rate*self.window), 1)
        )

    def get_model(self, num_classes, classifier_model=__pretrained_model_with_conv, audio_augmentations=(), spectrum_augmentations=()):
        model = Sequential()
        for augment in audio_augmentations:
            model.add(augment)
        model.add(self.get_melspectrogram())
        for augment in spectrum_augmentations:
            model.add(augment)
        model.add(classifier_model(self, num_classes))
        return model

    def load_file(self, audio_file):
        return self.file_loader.load(audio_file)

if __name__ == '__main__':
    import math
    import librosa
    import matplotlib.pyplot as plt

    # Create model builder
    builder = AudioModelBuilder(window=5, step=2.5)
    audio = builder.load_file(librosa.util.example('brahms'))

    builder.get_model(3).predict(audio)
    # Get melspectrogram layer from builder
    stft = builder.get_melspectrogram()
    # Predict melspectrogram of windowed items
    melspectrograms = stft.predict(audio)

    plt.figure()
    width = round(math.sqrt(melspectrograms.shape[0]))
    f, axarr = plt.subplots(width, int(math.ceil(melspectrograms.shape[0]/width)))
    for i, melspectrogram in enumerate(melspectrograms):
        axarr[i % width][i // width].imshow(melspectrogram.squeeze(), vmin=-50, vmax=5)
    plt.show()
