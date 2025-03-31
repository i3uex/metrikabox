from .data_loader import DataLoader
from .audio_loader import AudioLoader
from .encodec_loader import EncodecLoader

TYPE2LOADER = {
    "Audio": AudioLoader,
    "Encodec": EncodecLoader
}