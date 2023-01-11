import argparse
from config import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP
from predictor import AudioClassifier, AudioSegmenter

TASK2MODEL = {
    'classify': AudioClassifier,
    'segment': AudioSegmenter
}
parser = argparse.ArgumentParser(prog = 'AudioPredict', description = 'Predicts the input audio with the model provided')
parser.add_argument('filename')
parser.add_argument('model_id')
parser.add_argument('-t', '--task', default='classify', choices=TASK2MODEL.keys())
parser.add_argument('-sr', '--sample_rate', default=SAMPLE_RATE, type=int)
parser.add_argument('--window', default=CONTEXT_WINDOW, type=float)
parser.add_argument('--step', default=PROCESSING_STEP, type=float)
args = parser.parse_args()

print(TASK2MODEL[args.task](args.model_id, sample_rate=args.sample_rate, window=args.window, step=args.step).predict(args.filename))