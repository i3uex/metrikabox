from typing import List, Union

from pydub import AudioSegment
import tensorflow as tf
from keras.models import Sequential, Model
from keras import layers
import numpy as np

from augmentations.AugmentationLayer import AudioAugmentationLayer, SpectrogramAugmentationLayer
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP
from loaders import FileLoader
from model.classification import MNIST_convnet

# Tested for 1 step and 2 win @ 16000 Hz
DEFAULT_STFT_HOP = 256
DEFAULT_STFT_N_FFT = DEFAULT_STFT_HOP * 4
DEFAULT_STFT_WIN = DEFAULT_STFT_HOP * 4
DEFAULT_MEL_F_MIN = 0
DEFAULT_N_MELS = 128
DEFAULT_PREDEFINED_MODEL = MNIST_convnet()


class NormLayer(layers.Layer):

    def call(self, x, training=None, **kwargs):
        if x.dtype.is_integer:
            scale = 1. / float(1 << ((8 * x.dtype.size) - 1))
            # Rescale and format the data buffer
            return tf.math.scalar_mul(scale, tf.cast(x, dtype=tf.float32))
        return x


def get_classification_model(
        num_classes: int,
        input_shape: tuple,
        predefined_model: Model = DEFAULT_PREDEFINED_MODEL
) -> Model:
    """
    Get a classification model
    :param input_shape: Tuple of ints with the input shape
    :param num_classes: Desired number of classes
    :param predefined_model: Model to use as base for the classification model
    :return: Classification model
    """
    input_tensor = layers.Input(shape=input_shape)
    convolution_layer = layers.Conv2D(3, (3, 3), padding='same')(
        input_tensor)  # X has a dimension of (IMG_SIZE,N_MELS,3)
    base_model = predefined_model
    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
    base_model = base_model(convolution_layer)
    output_tensor = layers.Dense(num_classes if num_classes > 2 else 1,
                                 activation='softmax' if num_classes > 2 else 'sigmoid')(base_model)
    return Model(inputs=input_tensor, outputs=output_tensor)


class AudioModelBuilder:
    """
    Class to build audio models
    """

    def __init__(
            self,
            sample_rate: int = DEFAULT_SAMPLE_RATE,
            window: float = DEFAULT_WINDOW,
            step: float = DEFAULT_STEP,
            mel_f_min: int = DEFAULT_MEL_F_MIN,
            stft_nfft: int = DEFAULT_STFT_N_FFT,
            stft_window: int = DEFAULT_STFT_WIN,
            stft_hop: int = DEFAULT_STFT_HOP,
            stft_nmels: int = DEFAULT_N_MELS
    ):
        """
        Class to build audio models
        :param sample_rate: Desired sample rate
        :param window: Length of the window in seconds
        :param step: Length of the step in seconds
        :param mel_f_min: Min frequency of the mel filterbank
        :param stft_nfft: nfft for the stft
        :param stft_window: window for the stft
        :param stft_hop: hop for the stft
        :param stft_nmels: number of mels for the stft
        """
        self.sample_rate = sample_rate
        self.window = window
        self.step = step
        self.mel_f_min = mel_f_min
        self.stft_nfft = stft_nfft
        self.stft_window = stft_window
        self.stft_hop = stft_hop
        self.stft_nmels = stft_nmels
        self.file_loader = FileLoader(sample_rate=self.sample_rate, window=self.window, step=self.step)

    def get_melspectrogram(self) -> Model:
        """
        Get a melspectrogram layer
        :return: Layer to predict melspectrograms
        """
        return layers.MelSpectrogram(
            fft_length=self.stft_nfft,
            sequence_length=self.stft_window,
            sequence_stride=self.stft_hop,
            num_mel_bins=self.stft_nmels,
            sampling_rate=self.sample_rate,
            min_freq=self.mel_f_min,
            power_to_db=True
        )

    def get_model(
            self,
            num_classes: int,
            predefined_model: Model = None,
            audio_augmentations: List[AudioAugmentationLayer] = (),
            spectrum_augmentations: List[SpectrogramAugmentationLayer] = ()
        ) -> Model:
        """
        Get a model
        :param num_classes: number of classes to predict
        :param predefined_model: model to use as base for the classification model
        :param audio_augmentations: list of audio augmentations to apply
        :param spectrum_augmentations: list of spectrum augmentations to apply
        :return: Keras model
        """
        if not predefined_model:
            predefined_model = DEFAULT_PREDEFINED_MODEL
        model = Sequential()
        model.add(NormLayer())
        for augment in audio_augmentations:
            model.add(augment)
        melspectrogram_layer = self.get_melspectrogram()
        model.add(melspectrogram_layer)
        input_shape = (*melspectrogram_layer.compute_output_shape((int(self.sample_rate * self.window),)), 1)
        model.add(layers.Reshape(input_shape))
        for augment in spectrum_augmentations:
            model.add(augment)
        model.add(
            get_classification_model(
                num_classes,
                input_shape=input_shape,
                predefined_model=predefined_model
            )
        )
        return model

    def load_file(self, audio_file: Union[str, np.array, AudioSegment]) -> np.ndarray:
        return self.file_loader.load(audio_file)


if __name__ == '__main__':
    import math
    import librosa
    import matplotlib.pyplot as plt

    # Create model builder
    builder = AudioModelBuilder(window=5, step=2.5)
    audio = builder.load_file(librosa.util.example('brahms'))

    model = Sequential()
    # Normalize int16 to float32
    model.add(NormLayer())
    # Get melspectrogram layer from builder
    stft = builder.get_melspectrogram()
    model.add(stft)
    # Predict melspectrogram of windowed items
    melspectrograms = model.predict(audio)

    plt.figure()
    width = round(math.sqrt(melspectrograms.shape[0]))
    f, axarr = plt.subplots(width, int(math.ceil(melspectrograms.shape[0] / width)))
    for i, melspectrogram in enumerate(melspectrograms):
        axarr[i % width][i // width].imshow(melspectrogram.squeeze(), vmin=-50, vmax=5)
    plt.show()
