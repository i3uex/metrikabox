from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from kapre.composed import get_melspectrogram_layer

from config import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP
from loaders import FileLoader
from utils import get_mels_from_hop_and_win_lengths

DEFAULT_STFT_N_FFT = 1024
DEFAULT_STFT_WIN = 1024
DEFAULT_STFT_HOP = 256
DEFAULT_N_MELS = 128


class AudioModelBuilder:
    def __init__(self, sample_rate=SAMPLE_RATE, context_window=CONTEXT_WINDOW, processing_step=PROCESSING_STEP, stft_n_fft=DEFAULT_STFT_N_FFT, stft_window=DEFAULT_STFT_WIN, stft_hop=DEFAULT_STFT_HOP, n_mels=DEFAULT_N_MELS):
        self.sample_rate = sample_rate
        self.context_window = context_window
        self.processing_step = processing_step
        self.stft_n_fft = stft_n_fft
        self.stft_window = stft_window
        self.stft_hop = stft_hop
        self.n_mels = n_mels
        self.file_loader = FileLoader(sample_rate=self.sample_rate, window=self.context_window, step=self.processing_step)

    def __get_simple_MNIST_convnet(self, num_classes):
        model = Sequential()
        model.add(layers.Input(shape=(int(self.sample_rate*self.context_window), self.n_mels, 1)))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation="softmax"))
        return model

    def __pretrained_model_with_conv(self, num_classes, pretrained_model=MobileNetV3Small):
        input_tensor = layers.Input(shape=(get_mels_from_hop_and_win_lengths(self.stft_hop, self.stft_window, input_size=int(self.sample_rate*self.context_window)), self.n_mels, 1))
        convolution_layer = layers.Conv2D(3, (3, 3), padding='same')(input_tensor)  # X has a dimension of (IMG_SIZE,N_MELS,3)
        base_model = pretrained_model(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        for layer in base_model.layers:
            layer.trainable = True  # trainable has to be false in order to freeze the layers
        base_model = base_model(convolution_layer)
        op = layers.Dense(256, activation='relu')(base_model)
        op = layers.Dropout(.25)(op)
        output_tensor = layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax')(op)
        return Model(inputs=input_tensor, outputs=output_tensor)

    def get_melspectrogram(self):
        return get_melspectrogram_layer(
            n_fft=self.stft_n_fft,
            win_length=self.stft_window,
            hop_length=self.stft_hop,
            n_mels=self.n_mels,  # n_mels set to make squared image
            sample_rate=self.sample_rate,
            return_decibel=True,
            input_data_format='channels_last', output_data_format='channels_last',
            input_shape=(int(self.sample_rate*self.context_window), 1)
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
    builder = AudioModelBuilder(context_window=5, processing_step=2.5)
    audio = builder.load_file(librosa.util.example('brahms'))

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
