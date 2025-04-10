import keras


class AugmentationLayer(keras.layers.Layer):
    """
    Base class for augmentation layers
    """

    def compute_output_shape(self, input_shape):
        """
        Compute the output shape of the layer
        """
        # Return the input shape as the output shape
        return input_shape


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