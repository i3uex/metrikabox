import json
import gradio as gr
import soxr
from audio_classifier.utils import LOGGER
from audio_classifier import constants
from audio_classifier import AudioClassifier, AudioSegmenter


TASK2MODEL = {
    'classify': AudioClassifier,
    'segment': AudioSegmenter,
}


def infer(
        audio,
        model_path,
        model_config_path=None,
        task='segment'
):
    if not model_config_path:
        base_path, model_name = model_path.rsplit('.', 1)[0].rsplit('/', 1)
        model_config_path = f"{base_path}/{constants.MODEL_CONFIG_FOLDER}/{model_name}/model-config.json"
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
    return (predictions, "") if task == 'classify' else ("", json.dumps(predictions, default=str))


with gr.Blocks() as demo:
    gr.Markdown("""
    # MetrikaBox Inferencer
    This demo predicts the classes of an audio file using a pre-trained model and its configuration.
    """)
    with gr.Row():
        with gr.Column():
            inp = [
                gr.Audio(label="Audio file to infer"),
                gr.File(label="Model checkpoints (.keras)", file_count='single', type='filepath', file_types=[".keras"]),
                gr.File(label="Model configuration (.json)", file_count='single', type='filepath', file_types=[".json"]),
            ]
        with gr.Column():
            drop = gr.Dropdown(label="Prediction task", choices=list(TASK2MODEL.keys()))
            out = [
                gr.Label(label="Classification result"),
                gr.Textbox(label="Segmentation result", visible=False)
            ]
    inp.append(drop)
    btn = gr.Button("Classify")
    btn.click(fn=infer, inputs=inp, outputs=out)

    def update_visibility(prediction_type):  # Accept the event argument, even if not used
        if prediction_type == 'classify':
            return [gr.update(visible=True), gr.update(visible=False)]
        else:
            return [gr.update(visible=False), gr.update(visible=True)]

    def update_task(prediction_type):
        return gr.update(value=prediction_type.capitalize())

    drop.change(update_visibility, drop, out)
    drop.change(update_task, drop, btn)
demo.launch()
