import tensorflow as tf
from audio_classifier.augmentations.AugmentationLayer import AudioAugmentationLayer


class WhiteNoiseAugmentation(AudioAugmentationLayer):
    """
    Layer to apply white noise augmentation over the audio
    """
    def __init__(self, max_snr: float = 15., min_snr: float = 30, **kwargs):
        """
        Layer to apply white noise augmentation over the audio
        :param max_snr: maximum signal to noise ratio
        :param min_snr: minimum signal to noise ratio
        :param kwargs: other arguments
        """
        super(WhiteNoiseAugmentation, self).__init__(**kwargs)
        self.max_snr = max_snr
        self.min_snr = min_snr

    def call(self, sounds, training=None, **kwargs):
        """
        Apply white noise augmentation over the audio
        :param sounds: tensor with the audio to augment
        :param training: whether the model is training
        :param kwargs: other arguments
        :return:
        """
        if not training:
            return sounds
        sounds = tf.expand_dims(sounds, 2)
        snr = tf.random.uniform((tf.shape(sounds)[0], 1), self.min_snr, self.max_snr)
        rms_s = tf.math.sqrt(tf.math.reduce_mean(tf.math.pow(sounds, 2), axis=1))
        noise_std = tf.math.sqrt(rms_s**2/(10**(snr/10)))
        noise = tf.expand_dims(tf.random.normal((tf.shape(sounds)[0], sounds.shape[1]), mean=0.0, stddev=noise_std), 2)
        return tf.squeeze(sounds + noise, 2)

    def get_config(self):
        """
        Get the configuration of the layer
        :return: configuration of the layer
        """
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
    from audio_classifier.loaders import FileLoader
    from audio_classifier.model.builder import NormLayer
    SAMPLE_RATE = 16000
    audio = FileLoader(SAMPLE_RATE, 2, 1).load(librosa.util.example('brahms'))
    model = tf.keras.models.Sequential()
    model.add(NormLayer())
    model.add(WhiteNoiseAugmentation(max_snr=3, min_snr=6))
    a = model.predict(audio)
    print(a.shape)
    test_audio_index = 15
    sf.write('noised_file.wav', a[test_audio_index], SAMPLE_RATE, subtype='PCM_24')
    sf.write('original_file.wav', audio[test_audio_index], SAMPLE_RATE, subtype='PCM_24')
