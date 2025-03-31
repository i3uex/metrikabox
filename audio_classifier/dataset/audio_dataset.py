from audio_classifier import constants
from audio_classifier.dataset import Dataset
from audio_classifier.model import AudioModelBuilder
from audio_classifier.loaders import AudioLoader, FolderLoader


class AudioDataset(Dataset):
    def __init__(
            self,
            sample_rate: int = constants.DEFAULT_SAMPLE_RATE,
            stft_nfft: int = constants.DEFAULT_STFT_N_FFT,
            stft_win: int = constants.DEFAULT_STFT_WIN,
            stft_hop: int = constants.DEFAULT_STFT_HOP,
            stft_nmels: int = constants.DEFAULT_N_MELS,
            mel_f_min: int = constants.DEFAULT_MEL_F_MIN,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.stft_nfft = stft_nfft
        self.stft_win = stft_win
        self.stft_hop = stft_hop
        self.stft_nmels = stft_nmels
        self.mel_f_min = mel_f_min
        self.file_loader = AudioLoader(sample_rate=sample_rate, window=self.window, step=self.step)
        self.data_loader = FolderLoader(
            self.file_loader,
            class_loader=self.class_loader
        )
        self.model_builder = AudioModelBuilder

    def get_config(self):
        config = super().get_config()
        config.update({
            "sample_rate": self.sample_rate,
            "stft_nfft": self.stft_nfft,
            "stft_win": self.stft_win,
            "stft_hop": self.stft_hop,
            "stft_nmels": self.stft_nmels,
            "mel_f_min": self.mel_f_min
        })
        return config
