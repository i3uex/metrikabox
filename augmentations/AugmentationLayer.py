from tensorflow.keras.layers import Layer


class AugmentationLayer(Layer):
    """
    Base class for augmentation layers
    """
    pass


class AudioAugmentationLayer(AugmentationLayer):
    """
    Base class for audio augmentation layers
    """
    pass


class SpectrogramAugmentationLayer(AugmentationLayer):
    """
    Base class for spectrogram augmentation layers
    """
    pass