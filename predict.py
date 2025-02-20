import argparse
import json

from classes import AudioClassifier, AudioSegmenter

TASK2MODEL = {
    'classify': AudioClassifier,
    'segment': AudioSegmenter,
}
parser = argparse.ArgumentParser(prog = 'AudioPredict', description = 'Predicts the input audio with the model provided')
parser.add_argument('filename')
parser.add_argument('model_id')
parser.add_argument('-t', '--task', default='segment', choices=TASK2MODEL.keys())

if __name__ == '__main__':
    args = parser.parse_args()
    model = TASK2MODEL[args.task](args.model_id)
    base_file_name = args.filename.split(".", 1)[0]

    probabilities = model.predict_without_format(args.filename)
    with open(f"{base_file_name}_probas.json", 'w') as f:
        json.dump(probabilities, f)

    predictions = model.format_output(probabilities)
    with open(f"{base_file_name}.json", 'w') as f:
        json.dump(predictions, f)