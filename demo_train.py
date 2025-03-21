import datetime
import gradio as gr
from audio_classifier import Trainer, Dataset
from audio_classifier.config import DEFAULT_SAMPLE_RATE, CHECKPOINTS_FOLDER, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from audio_classifier.constants import AVAILABLE_KERAS_OPTIMIZERS, AVAILABLE_CLASS_LOADERS, AVAILABLE_MODELS, \
    AVAILABLE_AUDIO_AUGMENTATIONS, AVAILABLE_SPECTROGRAM_AUGMENTATIONS
from audio_classifier.model import DEFAULT_STFT_N_FFT, DEFAULT_STFT_WIN, DEFAULT_STFT_HOP, DEFAULT_N_MELS, \
    DEFAULT_MEL_F_MIN
from demo_utils import get_image_from_history


def train(
        folder,
        model,
        sample_rate,
        window,
        step,
        classes2avoid,
        checkpoints_folder,
        optimizer,
        batch_size,
        epochs,
        class_loader,
        learning_rate,
        model_id,
        stft_nfft,
        stft_win,
        stft_hop,
        stft_n_mels,
        mel_f_min,
        audio_augmentations,
        spectrogram_augmentations
):
    """
    Trains the model
    :param folder: Path to the older containing the audio files
    :param sample_rate: Sample rate the audios will be resampled to
    :param window: Seconds of audio to use
    :param step: Seconds to move the window
    :param classes2avoid: Classes to avoid from training model
    :param checkpoints_folder: Path to save the model checkpoints
    :param optimizer: String with the optimizer to use
    :param batch_size: Number of items to use in each batch
    :param epochs: Number of epochs to train the model
    :param class_loader: String with the class loader to use
    :param learning_rate: Learning rate for the optimizer
    :param model_id: String with the model id
    :param stft_nfft: Number of FFTs to use
    :param stft_win: Length of the STFT window
    :param stft_hop: Length of the STFT hop
    :param stft_n_mels: Number of mel bands to use
    :param mel_f_min: Minimum frequency for the mel bands
    :param model: Name of the predefined model to use. Any of keras.applications
    :param audio_augmentations: List of audio augmentations to use
    :param spectrogram_augmentations: List of spectrogram augmentations to use
    :return:
    """
    trainer = Trainer(
        stft_nfft=stft_nfft,
        stft_win=stft_win,
        stft_hop=stft_hop,
        stft_nmels=stft_n_mels,
        mel_f_min=mel_f_min,
        predefined_model=model,
        audio_augmentations=audio_augmentations,
        spectrogram_augmentations=spectrogram_augmentations
    )
    dataset = Dataset(
        folder,
        sample_rate=sample_rate,
        window=window,
        step=step,
        classes2avoid=classes2avoid.split(",") if classes2avoid else [],
        class_loader=class_loader
    )
    if not model_id:
        folder_name = folder.rsplit("/")[-1] if folder[-1] != "/" else folder.rsplit("/")[-2]
        model_id = f"{folder_name}_{str(datetime.datetime.now())}_{sample_rate}Hz_{window}w_{step}s"
    model_checkpoints, model_config_path, history = trainer.train(
        dataset,
        val_size=0.2,
        optimizer=optimizer,
        learning_rate=learning_rate,
        checkpoints_folder=checkpoints_folder,
        batch_size=batch_size,
        epochs=epochs,
        model_id=model_id,
    )
    return [model_checkpoints, model_config_path], [
        get_image_from_history(history.history, 'accuracy'),
        get_image_from_history(history.history, 'loss'),
        get_image_from_history(history.history, 'precision'),
        get_image_from_history(history.history, 'recall')
    ]


with gr.Blocks() as demo:
    gr.Markdown("""
    # MetrikaBox Trainer
    This demo trains a model using a dataset of audio files and a predefined model or a custom one.
    You can also set the training parameters like the optimizer, the learning rate, the batch size and other advanced parameters.
    """)
    inp = []
    with gr.Row():
        with gr.Column():
            inp.append(gr.Textbox(placeholder="/path/to/dataset", label="Dataset Path"))
            inp.append(gr.Dropdown(choices=sorted(AVAILABLE_MODELS.keys()), value="custom.MNIST_convnet", label="Model to train"))  # model
        with gr.Column():
            out = [
                gr.File(label="Model"),
                gr.Gallery(label="Training metrics")
            ]
    btn = gr.Button("Train")
    with gr.Accordion("Aditional training params", open=False):
        inp.extend([
            gr.Dropdown(
                label="Sampling rate",
                info="Sampling rate the audios will be converted to",
                choices=[8000, 16000, 22050, 32000, 44100],
                value=DEFAULT_SAMPLE_RATE
            ),  # sample_rate
            gr.Slider(
                label="Window",
                info="Seconds of audio to use for each item",
                minimum=1,
                maximum=10,
                value=2
            ),  # window
            gr.Slider(
                label="Step",
                info="Seconds to move the window for each item",
                minimum=1,
                maximum=10,
                value=1
            ),  # step
            gr.Text(
                label="Classes to avoid",
                info="The classes of the dataset that will be omitted separated by \",\""
            ),  # classes2avoid
            gr.Text(
                label="Checkpoints folder",
                info="Folder in which the checkpoints of the model will be saved to",
                value=CHECKPOINTS_FOLDER
            ),  # checkpoints_folder
            gr.Dropdown(
                label="Optimizer",
                info="Optimizer to be used in training",
                choices=sorted(AVAILABLE_KERAS_OPTIMIZERS.keys()),
                value="Adam"
            ),  # optimizer
            gr.Number(
                label="Batch Size",
                info="Number of items to use in each batch",
                value=DEFAULT_BATCH_SIZE
            ),  # batch_size
            gr.Number(
                label="Epochs",
                info="Number of epochs to train the model",
                value=DEFAULT_EPOCHS
            ),  # epochs
            gr.Dropdown(
                label="Class loader",
                info="Class loader to use for the dataset",
                choices=sorted(AVAILABLE_CLASS_LOADERS.keys()),
                value="ClassLoaderFromFolderName"
            ),  # class_loader
            gr.Slider(
                label="Learning rate",
                info="Learning rate for the optimizer",
                minimum=1.e-6,
                maximum=0.1,
                value=0.001
            ),  # learning_rate
            gr.Text(
                label="Model ID",
                info="ID to use for the model",
                value=None
            ),  # model_id
            gr.Number(
                label="STFT number FFT",
                info="Number of FFTs to use",
                value=DEFAULT_STFT_N_FFT
            ),  # stft_nfft
            gr.Number(
                label="STFT window",
                info="Length of the STFT window",
                value=DEFAULT_STFT_WIN
            ),  # stft_win
            gr.Number(
                label="STFT hop",
                info="Length of the STFT hop",
                value=DEFAULT_STFT_HOP
            ),  # stft_hop
            gr.Number(
                label="Mel bands",
                info="Number of mel bands to use",
                value=DEFAULT_N_MELS
            ),  # stft_nmels
            gr.Number(
                label="Minimum frequency",
                info="Minimum frequency for the mel bands",
                value=DEFAULT_MEL_F_MIN
            ),  # mel_f_min
            gr.Dropdown(
                label="Audio augmentations",
                info="List of audio augmentations to use",
                choices=sorted(AVAILABLE_AUDIO_AUGMENTATIONS.keys()),
                multiselect=True
            ),  # audio_augmentations
            gr.Dropdown(
                label="Spectrogram augmentations",
                choices=sorted(AVAILABLE_SPECTROGRAM_AUGMENTATIONS.keys()),
                multiselect=True
            ),  # spectrogram_augmentations
        ])
    btn.click(fn=train, inputs=inp, outputs=out)

demo.launch()

