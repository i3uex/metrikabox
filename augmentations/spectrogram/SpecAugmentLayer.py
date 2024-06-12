from augmentations.AugmentationLayer import SpectrogramAugmentationLayer
from kapre.augmentation import SpecAugment


class SpecAugmentLayer(SpectrogramAugmentationLayer, SpecAugment):
    """
    Layer to apply SpecAugment from Kapre over the spectrogram
    """
    pass
