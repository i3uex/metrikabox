import json
import gradio as gr
import soxr
from audio_classifier import AudioClassifier, AudioSegmenter
from audio_classifier.config import MODEL_CONFIG_FOLDER
from audio_classifier.utils import LOGGER


TASK2MODEL = {
    'classify': AudioClassifier,
    'segment': AudioSegmenter,
}


def predict(
        audio,
        model_path,
        model_config_path=None,
        task='segment'
):
    if not model_config_path:
        base_path, model_name = model_path.rsplit('.', 1)[0].rsplit('/', 1)
        model_config_path = f"{base_path}/{MODEL_CONFIG_FOLDER}/{model_name}/model-config.json"
    sr, audio = audio
    model = TASK2MODEL[task](model_path, model_config_path)
    model_sample_rate = model.model_config.get('sample_rate')
    if len(audio.shape) > 1:
        LOGGER.warning(f"Channels of the audio {audio.shape[-1]} does not match the channels of the model {1}. Downmixing.")
        audio = audio.mean(1)
    if sr != model_sample_rate:
        LOGGER.warning(f"Sample rate of the audio {sr} does not match the sample rate of the model {model_sample_rate}. Resampling.")
        audio = soxr.resample(audio, sr, model_sample_rate)
    predictions = model.predict(audio)
    return json.dumps(predictions, default=str)


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(),
        gr.File(file_count='single', type='filepath'),
        gr.File(file_count='single', type='filepath'),
        gr.Dropdown(choices=list(TASK2MODEL.keys())),
    ],
    outputs=["text"],
)

demo.launch()
