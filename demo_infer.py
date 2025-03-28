import json
import gradio as gr
import soxr
from audio_classifier.utils import LOGGER
from audio_classifier import constants
from audio_classifier.infer import TASK2MODEL
from audio_classifier.loaders.data_loaders import AudioLoader, EncodecLoader

TYPE2LOADER = {
    "Audio": AudioLoader,
    "Encodec": EncodecLoader
}

LOADER_TYPES = sorted(TYPE2LOADER.keys())


def infer(
        audio,
        encodec,
        model_path,
        model_config_path=None,
        loader_type='Audio',
        task='segment',
):
    if not model_config_path:
        base_path, model_name = model_path.rsplit('.', 1)[0].rsplit('/', 1)
        model_config_path = f"{base_path}/{constants.MODEL_CONFIG_FOLDER}/{model_name}/model-config.json"
    if loader_type == 'Audio':
        sr, audio = audio
        model = TASK2MODEL[task](model_path, model_config_path, TYPE2LOADER['Audio'])
        model_sample_rate = model.model_config.get('sample_rate')
        if len(audio.shape) > 1:
            LOGGER.warning(f"Channels of the audio {audio.shape[-1]} does not match the channels of the model {1}. Downmixing.")
            audio = audio.mean(1)
        if sr != model_sample_rate:
            LOGGER.warning(f"Sample rate of the audio {sr} does not match the sample rate of the model {model_sample_rate}. Resampling.")
            audio = soxr.resample(audio, sr, model_sample_rate)
    else:
        model = TASK2MODEL[task](model_path, model_config_path, TYPE2LOADER['Encodec'])
        audio = encodec
    predictions = model.predict(audio)
    return (predictions, "") if task == 'classify' else ("", json.dumps(predictions, default=str))


with gr.Blocks() as demo:
    gr.Markdown("""
    # MetrikaBox Inferencer
    This demo predicts the classes of an audio file using a pre-trained model and its configuration.
    """)
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Audio file to infer")
            encodec = gr.File(label="Encodec file to infer", visible=False, file_types=[".ecdc"])
            inp = [
                audio,
                encodec,
                gr.File(label="Model checkpoints (.keras)", file_count='single', type='filepath', file_types=[".keras"]),
                gr.File(label="Model configuration (.json)", file_count='single', type='filepath', file_types=[".json"]),
            ]
            data_type_dropdown = gr.Dropdown(
                    label="Dataset format",
                    info="Format of files in the dataset",
                    choices=LOADER_TYPES,
                    value=LOADER_TYPES[0]
                )
            inp.append(data_type_dropdown)

            def update_visibility(selected_type, expected_type):  # Accept the event argument, even if not used
                if selected_type == expected_type:
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            data_type_dropdown.change(lambda x: update_visibility(x, "Audio"), data_type_dropdown, audio)
            data_type_dropdown.change(lambda x: update_visibility(x, "Encodec"), data_type_dropdown, encodec)
        with gr.Column():
            drop = gr.Dropdown(label="Prediction task", choices=list(TASK2MODEL.keys()))
            classification_result = gr.Label(label="Classification result")
            segmentation_result = gr.Label(label="Segmentation result", visible=False)
            out = [classification_result, segmentation_result]
    inp.append(drop)
    btn = gr.Button("Classify")
    btn.click(fn=infer, inputs=inp, outputs=out)
    drop.change(lambda x: update_visibility(x, "classify"), drop, classification_result)
    drop.change(lambda x: update_visibility(x, "segment"), drop, segmentation_result)

    def update_task_name(prediction_type):
        return gr.update(value=prediction_type.capitalize())
    drop.change(update_task_name, drop, btn)
demo.launch()
