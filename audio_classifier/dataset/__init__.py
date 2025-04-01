from .dataset import Dataset
from .audio_dataset import AudioDataset
from .encodec_dataset import EncodecDataset

TYPE2DATASET = {
    "Audio": AudioDataset,
    "EnCodec": EncodecDataset
}
