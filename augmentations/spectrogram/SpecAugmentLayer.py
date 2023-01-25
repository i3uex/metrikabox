from augmentations.AugmentationLayer import SpectrogramAugmentationLayer
from kapre.augmentation import SpecAugment

class SpecAugmentLayer(SpectrogramAugmentationLayer, SpecAugment):
    pass
