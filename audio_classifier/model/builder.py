from abc import abstractmethod, ABC
from typing import List
import keras
import tensorflow as tf
from audio_classifier import constants
from audio_classifier.loaders import FileLoader
from audio_classifier.model.classification import MNIST_convnet
from audio_classifier.augmentations import AudioAugmentationLayer, SpectrogramAugmentationLayer

DEFAULT_PREDEFINED_MODEL = MNIST_convnet()


@keras.saving.register_keras_serializable()
class NormLayer(keras.layers.Layer):

    def call(self, x, training=None, **kwargs):
        if x.dtype.is_integer:
            scale = 1. / float(1 << ((8 * x.dtype.size) - 1))
            # Rescale and format the data buffer
            return tf.math.scalar_mul(scale, tf.cast(x, dtype=tf.float32))
        return x

    def get_config(self):
        return super().get_config()


def get_classification_model(
        num_classes: int,
        predefined_model: keras.models.Model = DEFAULT_PREDEFINED_MODEL
) -> keras.models.Model:
    """
    Get a classification model
    :param input_shape: Tuple of ints with the input shape
    :param num_classes: Desired number of classes
    :param predefined_model: Model to use as base for the classification model
    :return: Classification model
    """
    model = keras.models.Sequential()
    for layer in predefined_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
    model.add(predefined_model)
    model.add(keras.layers.Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid'))
    return model


class ModelBuilder(ABC):

    def __init__(self, window: float = constants.DEFAULT_WINDOW, step: float = constants.DEFAULT_STEP, **kwargs):
        self.window = window
        self.step = step


    @abstractmethod
    def get_preprocessing_layer(
            self,
            audio_augmentations: List[AudioAugmentationLayer] = (),
            spectrum_augmentations: List[SpectrogramAugmentationLayer] = ()
        ):
        return keras.models.Sequential()

    def get_model(
            self,
            num_classes: int,
            predefined_model: keras.models.Model = None,
            **kwargs
    ) -> keras.models.Model:
        """
        Get a model
        :param num_classes: number of classes to predict
        :param predefined_model: model to use as base for the classification model
        :param audio_augmentations: list of audio augmentations to apply
        :param spectrum_augmentations: list of spectrum augmentations to apply
        :return: Keras model
        """
        model = keras.models.Sequential()
        model.add(self.get_preprocessing_layer(**kwargs))
        model.add(
            get_classification_model(
                num_classes,
                predefined_model=predefined_model
            )
        )
        return model


class AudioModelBuilder(ModelBuilder):
    """
    Class to build audio models
    """

    def __init__(
            self,
            sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
            mel_f_min: int = constants.DEFAULT_MEL_F_MIN,
            stft_nfft: int = constants.DEFAULT_STFT_N_FFT,
            stft_window: int = constants.DEFAULT_STFT_WIN,
            stft_hop: int = constants.DEFAULT_STFT_HOP,
            stft_nmels: int = constants.DEFAULT_N_MELS,
            **kwargs
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
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.mel_f_min = mel_f_min
        self.stft_nfft = stft_nfft
        self.stft_window = stft_window
        self.stft_hop = stft_hop
        self.stft_nmels = stft_nmels

    def get_melspectrogram(self) -> keras.Layer:
        """
        Get a melspectrogram layer
        :return: Layer to predict melspectrograms
        """
        return keras.layers.MelSpectrogram(
            fft_length=self.stft_nfft,
            sequence_length=self.stft_window,
            sequence_stride=self.stft_hop,
            num_mel_bins=self.stft_nmels,
            sampling_rate=self.sample_rate,
            min_freq=self.mel_f_min,
            power_to_db=False
        )

    def get_preprocessing_layer(
            self,
            audio_augmentations: List[AudioAugmentationLayer] = (),
            spectrum_augmentations: List[SpectrogramAugmentationLayer] = ()
        ):
        model = super().get_preprocessing_layer(audio_augmentations, spectrum_augmentations)
        model.add(NormLayer())
        for augment in audio_augmentations:
            model.add(augment)
        melspectrogram_layer = self.get_melspectrogram()
        model.add(melspectrogram_layer)
        input_shape = (*melspectrogram_layer.compute_output_shape((int(self.sample_rate * self.window),)), 1)
        model.add(keras.layers.Reshape(input_shape))
        for augment in spectrum_augmentations:
            model.add(augment)
        model.add(keras.layers.Conv2D(3, 1, padding='same'))
        return model


class EncodecModelBuilder(ModelBuilder):
    """
    Class to build audio models
    """

    def __init__(
            self,
            model: str = 'encodec_24khz',
            decode: bool = True,
            expected_codebooks: int = 8,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.decode = decode
        self.expected_codebooks = expected_codebooks

    def get_preprocessing_layer(
            self,
            audio_augmentations: List[AudioAugmentationLayer] = (),
            spectrum_augmentations: List[SpectrogramAugmentationLayer] = ()
        ):
        model = super().get_preprocessing_layer(audio_augmentations, spectrum_augmentations)
        # Scale up to 32 if not decoding (will scale up to 128) and codebooks are less than 32
        if not self.decode and self.expected_codebooks < 32:
            model.add(keras.layers.Conv1D(32, 1, padding='same'))
        model.add(keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, -1)))
        model.add(keras.layers.Conv2D(3, 1, padding='same'))
        return model


if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    step = 2.5
    window = 5
    sample_rate = 16000
    fl = FileLoader(sample_rate=sample_rate, window=window, step=step)
    windowed_audio = fl.load("../example.ogg", max_duration=10)
    # Create model builder
    builder = AudioModelBuilder(sample_rate=sample_rate, window=window, step=step)

    model = keras.models.Sequential()
    # Normalize int16 to float32
    model.add(NormLayer())
    # Get melspectrogram layer from builder
    stft = builder.get_melspectrogram()
    model.add(stft)
    # Predict melspectrogram of windowed items
    melspectrograms = model.predict(windowed_audio)

    plt.figure()
    width = round(math.sqrt(melspectrograms.shape[0]))
    f, axarr = plt.subplots(width, int(math.ceil(melspectrograms.shape[0] / width)))
    for i, melspectrogram in enumerate(melspectrograms):
        axarr[i % width][i // width].imshow(melspectrogram.squeeze(), vmin=-50, vmax=5)
    plt.show()
