import numpy as np
from audio_classifier.predictor.audio_model import AudioModel


class AudioClassifier(AudioModel):
    """
    Class to predict audio classes
    """
    def format_output(self, y: np.ndarray, fn=np.mean):
        """
        Format the output of the model
        :param fn: function to apply to ensamble the predictions
        :param y: class predictions
        :return:
        """
        ensemble = fn(y.T, axis=1)
        if len(ensemble) == 1:
            ensemble = np.insert(ensemble, 0, 1 - ensemble[0])
        return {self.encoder.classes_[i]: sc for i, sc in enumerate(ensemble)}


if __name__ == '__main__':
    import sys
    print(AudioClassifier(sys.argv[2]).predict(sys.argv[1]))