from .data_model import DataModel
from .data_classifier import DataClassifier
from .data_segmenter import DataSegmenter
TASK2MODEL = {
    'classify': DataClassifier,
    'segment': DataSegmenter,
}