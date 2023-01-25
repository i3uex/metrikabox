import tensorflow as tf
from augmentations.AugmentationLayer import AudioAugmentationLayer


class WhiteNoiseAugmentation(AudioAugmentationLayer):
    def __init__(self, max_snr=15, min_snr=30, **kwargs):
        super(WhiteNoiseAugmentation, self).__init__(**kwargs)
        self.max_snr = max_snr
        self.min_snr = min_snr

    def call(self, sounds, training=None, **kwargs):
        if not training:
            return sounds
        snr = tf.random.uniform((tf.shape(sounds)[0], 1), self.min_snr, self.max_snr)
        rms_s = tf.math.sqrt(tf.math.reduce_mean(tf.math.pow(sounds, 2), axis=1))
        noise_std = tf.math.sqrt(rms_s**2/(10**(snr/10)))
        noise = tf.expand_dims(tf.random.normal((tf.shape(sounds)[0], sounds.shape[1]), mean=0.0, stddev=noise_std), 2)
        return sounds + noise

    def get_config(self):
        config = super(WhiteNoiseAugmentation, self).get_config()
        config.update(
            {
                'max_snr': self.max_snr,
                'min_snr': self.min_snr
            }
        )
        return config

if __name__ == '__main__':
    import librosa
    import soundfile as sf
    from utils import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP
    from loaders import FileLoader
    audio = FileLoader(SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP).load(librosa.util.example('brahms'))
    model = tf.keras.models.Sequential()
    model.add(WhiteNoiseAugmentation())
    a = model.predict(audio)
    test_audio_index = 15
    sf.write('../noised_file.wav', a[test_audio_index], SAMPLE_RATE, subtype='PCM_24')
    sf.write('../original_file.wav', audio[test_audio_index], SAMPLE_RATE, subtype='PCM_24')
