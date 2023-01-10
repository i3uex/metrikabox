import numpy as np
from predictor.audio_model import AudioModel

class AudioClassifier(AudioModel):
    def _format_output(self, y):
        mean_detections = np.mean(y.T, axis=1)
        score = mean_detections.max()
        return self.encoder.inverse_transform(np.expand_dims(mean_detections, 0))[0], score

if __name__ == '__main__':
    import sys
    print(AudioClassifier(sys.argv[2]).predict(sys.argv[1]))