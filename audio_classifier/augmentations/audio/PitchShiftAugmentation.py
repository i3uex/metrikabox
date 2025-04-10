import random
import tensorflow as tf
from audio_classifier.augmentations.AugmentationLayer import AudioAugmentationLayer


class PitchShiftAugmentation(AudioAugmentationLayer):
    """PitchShift

    Basic pitch shifter which computes fft, shifts and ifft
    """

    def __init__(self, shift: int = 750, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = shift

    def call(self, sounds, training=None):
        if training:
            return sounds

        def _pitch_shift(single_audio):
            _shift = int(random.uniform(-self.shift, self.shift))

            r_fft = tf.signal.rfft(single_audio)
            r_fft = tf.roll(r_fft, _shift, axis=0)
            zeros = tf.complex(tf.zeros([tf.abs(_shift)]), tf.zeros([tf.abs(_shift)]))

            if _shift < 0:
                r_fft = tf.concat([r_fft[:_shift], zeros], axis=0)
            else:
                r_fft = tf.concat([zeros, r_fft[_shift:]], axis=0)
            return tf.signal.irfft(r_fft)

        return tf.map_fn(_pitch_shift, sounds)

    def get_config(self):
        """
        Get the configuration of the layer
        :return: configuration of the layer
        """
        config = super(PitchShiftAugmentation, self).get_config()
        config.update({'shift': self.shift})
        return config


if __name__ == '__main__':
    import librosa
    import numpy as np
    import soundfile as sf
    from audio_classifier.loaders.data_loaders import AudioLoader
    from audio_classifier.model.builder import NormLayer
    SAMPLE_RATE = 16000
    test_shift = 500
    audio = AudioLoader(SAMPLE_RATE, window=2, step=1).load(librosa.util.example('libri1'))
    model = tf.keras.models.Sequential()
    model.add(NormLayer())
    model.add(PitchShiftAugmentation(shift=test_shift))
    a = model.predict(audio)
    model = tf.keras.models.Sequential()
    model.add(NormLayer())
    model.add(PitchShiftAugmentation(shift=-test_shift))
    b = model.predict(audio)
    print(a.shape)
    test_audio_index = 10
    sf.write('pitched_up_file.wav', np.array(a[test_audio_index], dtype=np.int16), SAMPLE_RATE, subtype='PCM_16')
    sf.write('pitched_down_file.wav', np.array(b[test_audio_index], dtype=np.int16), SAMPLE_RATE, subtype='PCM_16')
    sf.write('original_file.wav', np.array(audio[test_audio_index], dtype=np.int16), SAMPLE_RATE, subtype='PCM_16')