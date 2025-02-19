import numpy as np
from classes.predictor.audio_model import AudioModel


class AudioClassifier(AudioModel):
    """
    Class to predict audio classes
    """
    def format_output(self, y: np.ndarray):
        """
        Format the output of the model
        :param y: class predictions
        :return:
        """
        mean_detections = np.mean(y.T, axis=1)
        score = mean_detections.max()
        if len(self.encoder.classes_) == 2 and score < 0.5:
            score = 1-score
        return self.encoder.inverse_transform(np.expand_dims(mean_detections, 0))[0], score


if __name__ == '__main__':
    import sys
    print(AudioClassifier(sys.argv[2]).predict(sys.argv[1]))