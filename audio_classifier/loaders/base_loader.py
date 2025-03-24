import numpy as np
from .. import constants

class BaseLoader:
    """
    Base class to load audio files
    """
    def __init__(self, window: float = constants.DEFAULT_WINDOW, step: float = constants.DEFAULT_STEP):
        self.window = window
        self.step = step

    def load(self, audio: str) -> np.ndarray:
        pass
