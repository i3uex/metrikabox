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
        ensamble = fn(y.T, axis=1)
        score = ensamble.max()
        if len(self.encoder.classes_) == 2 and score < 0.5:
            score = 1-score
        return {self.encoder.inverse_transform(np.expand_dims(ensamble, 0))[0]: score}


if __name__ == '__main__':
    import sys
    print(AudioClassifier(sys.argv[2]).predict(sys.argv[1]))