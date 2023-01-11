import argparse
from predictor import AudioClassifier, AudioSegmenter

TASK2MODEL = {
    'classify': AudioClassifier,
    'segment': AudioSegmenter
}
parser = argparse.ArgumentParser(prog = 'AudioPredict', description = 'Predicts the input audio with the model provided')
parser.add_argument('filename')
parser.add_argument('model_id')
parser.add_argument('-t', '--task', default='classify', choices=TASK2MODEL.keys())
args = parser.parse_args()
print(TASK2MODEL[args.task](args.model_id).predict(args.filename))