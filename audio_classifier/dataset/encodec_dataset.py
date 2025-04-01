from audio_classifier.dataset import Dataset
from audio_classifier.loaders import FolderLoader
from audio_classifier.loaders.data_loaders import EncodecLoader
from audio_classifier.model import EncodecModelBuilder


class EncodecDataset(Dataset):
    def __init__(
            self,
            model: str = 'encodec_24khz',
            decode: bool = True,
            bandwidth: int = 6.,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.file_loader = EncodecLoader(model=model, decode=decode, bandwidth=bandwidth)
        self.data_loader = FolderLoader(
            self.file_loader,
            class_loader=self.class_loader,
            audio_formats=['.ecdc']
        )
        self.model_builder = EncodecModelBuilder

    def get_config(self):
        config = super().get_config()
        config.update({
            "model": self.file_loader.model_name,
            "decode": self.file_loader.decode,
            "bandwidth": self.file_loader.bandwidth,
            "frame_rate": self.file_loader.frame_rate,
            "expected_codebooks": self.file_loader.codebooks
        })
        return config



