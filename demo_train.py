import gradio as gr
import numpy as np
from audio_classifier import constants
from audio_classifier.dataset import TYPE2DATASET
from demo_utils import get_image_from_history
from main import train as _train

DATASET_TYPES = sorted(TYPE2DATASET.keys())


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
        spectrogram_augmentations,
        reduce_lr_on_plateau_patience,
        early_stopping_patience,
        early_stopping_metric,
        checkpoint_metric,
        encodec_model,
        encodec_decode,
        bandwidth,
        dataset_type
):
    """
    Trains the model
    :param bandwidth: Bandwidth the audio was encoded to (Only use with 'EnCodec' dataset type)
    :param encodec_decode: Whether if the audio should be decoded. Increases latent space (Only use with 'EnCodec' dataset type)
    :param encodec_model: EnCodec Model the audios where encoded with (Only use with 'EnCodec' dataset type)
    :param dataset_type: Format of files in the dataset
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
    :param reduce_lr_on_plateau_patience: Patience for reducing the learning rate on plateau (0 for no reducing)
    :param early_stopping_patience: Patience for early stopping (0 for no early stopping)
    :param early_stopping_metric: Metric used to monitor the early stopping
    :param checkpoint_metric: Metric used to obtain the best checkpoint of the model
    :return:
    """
    model_checkpoints, model_config_path, history = _train(
        folder=folder,
        model=model,
        sample_rate=sample_rate,
        window=window,
        step=step,
        classes2avoid=classes2avoid,
        checkpoints_folder=checkpoints_folder,
        optimizer=optimizer,
        batch_size=batch_size,
        epochs=epochs,
        class_loader=class_loader,
        learning_rate=learning_rate,
        model_id=model_id,
        stft_nfft=stft_nfft,
        stft_win=stft_win,
        stft_hop=stft_hop,
        stft_nmels=stft_n_mels,
        mel_f_min=mel_f_min,
        audio_augmentations=audio_augmentations,
        spectrogram_augmentations=spectrogram_augmentations,
        reduce_lr_on_plateau_patience=reduce_lr_on_plateau_patience,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        checkpoint_metric=checkpoint_metric,
        encodec_model=encodec_model,
        encodec_decode=encodec_decode,
        bandwidth=bandwidth,
        dataset_type=dataset_type
    )
    return [model_checkpoints, model_config_path], [
        (get_image_from_history(history.history, 'accuracy'), "Model Accuracy"),
        (get_image_from_history(history.history, 'loss', best_metric_fn=np.argmin, percentage=False), "Model Loss"),
        (get_image_from_history(history.history, 'precision'), "Model Precision"),
        (get_image_from_history(history.history, 'recall'), "Model Recall")
    ]


def main():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # MetrikaBox Trainer
        This demo trains a model using a dataset of audio files and a predefined model or a custom one.
        You can also set the training parameters like the optimizer, the learning rate, the batch size and other advanced parameters.
        """)
        inp = []
        with gr.Row():
            with gr.Column():
                inp.append(gr.Textbox(
                    placeholder="/path/to/dataset",
                    info="Path to the directory containing the folders with the class labels. Each folder must contain the audio files of a single class",
                    label="Dataset Path"
                ))
                dataset_type = gr.Dropdown(
                    label="Dataset format",
                    info="Format of files in the dataset",
                    choices=DATASET_TYPES,
                    value=DATASET_TYPES[0]
                )
                inp.append(gr.Dropdown(
                    label="Model to train",
                    info="Predefined model to train with the selected dataset",
                    choices=sorted(constants.AVAILABLE_MODELS.keys()),
                    value=constants.DEFAULT_MODEL,
                ))  # model
            with gr.Column():
                out = [
                    gr.File(label="Model"),
                    gr.Gallery(label="Training metrics")
                ]
        btn = gr.Button("Train")
        with gr.Accordion("Aditional training params", open=False):
            with gr.Accordion("Model params", open=False):
                with gr.Row():
                    window = gr.Slider(
                        label="Window",
                        info="Seconds of audio to use for each item",
                        minimum=1,
                        maximum=10,
                        value=constants.DEFAULT_WINDOW
                    )
                    step = gr.Slider(
                        label="Step",
                        info="Seconds to move the window for each item",
                        minimum=1,
                        maximum=10,
                        value=constants.DEFAULT_STEP
                    )

            with gr.Accordion("Audio params", open=False) as a:
                with gr.Row():
                    sample_rate = gr.Dropdown(
                        label="Sampling rate",
                        info="Sampling rate the audios will be converted to",
                        choices=[8000, 16000, 22050, 32000, 44100],
                        value=constants.DEFAULT_SAMPLE_RATE
                    )
                    stft_nfft = gr.Number(
                        label="STFT number FFT",
                        info="Number of FFTs to use",
                        value=constants.DEFAULT_STFT_N_FFT
                    )
                    stft_win = gr.Number(
                        label="STFT window",
                        info="Length of the STFT window",
                        value=constants.DEFAULT_STFT_WIN
                    )
                    stft_hop = gr.Number(
                        label="STFT hop",
                        info="Length of the STFT hop",
                        value=constants.DEFAULT_STFT_HOP
                    )
                    stft_n_mels = gr.Number(
                        label="Mel bands",
                        info="Number of mel bands to use",
                        value=constants.DEFAULT_N_MELS
                    )
                    mel_f_min = gr.Number(
                        label="Minimum frequency",
                        info="Minimum frequency for the mel bands",
                        value=constants.DEFAULT_MEL_F_MIN
                    )

                def update_audio_params_visibility(type):  # Accept the event argument, even if not used
                    if type == 'Audio':
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                dataset_type.change(update_audio_params_visibility, dataset_type, a)

            with gr.Accordion("Encodec params", open=False, visible=False) as a:
                with gr.Row():
                    encodec_model = gr.Dropdown(
                        label="Encodec model",
                        info="Model the audios where encoded with",
                        choices=['encodec_24khz', 'encodec_48khz'],
                        value='encodec_24khz'
                    )
                    encodec_decode = gr.Checkbox(
                        label="Decode",
                        info="Whether if the audio should be decoded. Increases latent space",
                        value=True
                    )
                    bandwidth = gr.Dropdown(
                        label="Bandwidth",
                        info="Bandwidth the audio was encoded to",
                        choices=[1.5, 3.0, 6.0, 12.0, 24.0],
                        value=6.0
                    )
                def update_encodec_params_visibility(type):  # Accept the event argument, even if not used
                    if type == 'Encodec':
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                dataset_type.change(update_encodec_params_visibility, dataset_type, a)

            with gr.Accordion("Dataset params", open=False):
                with gr.Row():
                    class_loader = gr.Dropdown(
                        label="Class loader",
                        info="Class loader to use for the dataset",
                        choices=sorted(constants.AVAILABLE_CLASS_LOADERS.keys()),
                        value=constants.DEFAULT_CLASS_LOADER
                    )
                    classes2avoid = gr.Text(
                        label="Classes to avoid",
                        info="The classes of the dataset that will be omitted separated by \",\""
                    )
            with gr.Accordion("Training hyperparams", open=False):
                model_id = gr.Text(
                    label="Model ID",
                    info="ID to use for the model",
                    value=None,
                    placeholder=constants.DEFAULT_MODEL_ID
                )
                with gr.Row():
                    optimizer = gr.Dropdown(
                        label="Optimizer",
                        info="Optimizer to be used in training",
                        choices=sorted(constants.AVAILABLE_KERAS_OPTIMIZERS.keys()),
                        value=constants.DEFAULT_OPTIMIZER
                    )
                    batch_size = gr.Number(
                        label="Batch Size",
                        info="Number of items to use in each batch",
                        value=constants.DEFAULT_BATCH_SIZE
                    )
                    epochs = gr.Number(
                        label="Epochs",
                        info="Number of epochs to train the model",
                        value=constants.DEFAULT_EPOCHS
                    )
                    learning_rate = gr.Slider(
                        label="Learning rate",
                        info="Learning rate for the optimizer",
                        minimum=1.e-6,
                        maximum=0.1,
                        value=constants.DEFAULT_LR
                    )
                    checkpoint_metric = gr.Dropdown(
                        label="Checkpoint metric",
                        info="Metric used to obtain the best checkpoint of the model",
                        choices=["accuracy", "val_accuracy", "loss", "val_loss", "precision", "val_precision", "recall",
                                 "val_recall"],
                        value=constants.DEFAULT_CHECKPOINT_METRIC
                    )
                    checkpoints_folder = gr.Text(
                        label="Checkpoints folder",
                        info="Folder in which the checkpoints of the model will be saved to",
                        value=constants.CHECKPOINTS_FOLDER
                    )
            with gr.Accordion("Data augmentation params", open=False):
                with gr.Row():
                    audio_augmentations = gr.Dropdown(
                        label="Audio augmentations",
                        info="List of audio augmentations to use",
                        choices=sorted(constants.AVAILABLE_AUDIO_AUGMENTATIONS.keys()),
                        multiselect=True
                    )
                    spectrogram_augmentations = gr.Dropdown(
                        label="Spectrogram augmentations",
                        info="List of spectrogram augmentations to use",
                        choices=sorted(constants.AVAILABLE_SPECTROGRAM_AUGMENTATIONS.keys()),
                        multiselect=True
                    )

            with gr.Accordion("Aditional training params", open=False):
                reduce_lr_patience = gr.Slider(
                    label="Reduce LR on Plateau patience",
                    info="Patience for reducing the learning rate on plateau (0 for no reducing)",
                    minimum=0,
                    maximum=100,
                    value=constants.DEFAULT_REDUCE_LR_ON_PLATEAU_PATIENCE
                )
                with gr.Row():
                    early_stopping_patience = gr.Slider(
                        label="Early Stopping patience",
                        info="Patience for early stopping (0 for no early stopping)",
                        minimum=0,
                        maximum=100,
                        value=constants.DEFAULT_EARLY_STOPPING_PATIENCE
                    )
                    early_stopping_metric = gr.Dropdown(
                        label="Early Stopping metric",
                        info="Metric used to monitor the early stopping",
                        choices=["accuracy", "val_accuracy", "loss", "val_loss", "precision", "val_precision", "recall",
                                 "val_recall"],
                        value=constants.DEFAULT_EARLY_STOPPING_METRIC
                    )
            inp.extend([
                sample_rate, window, step, classes2avoid, checkpoints_folder, optimizer, batch_size, epochs,
                class_loader, learning_rate, model_id, stft_nfft, stft_win, stft_hop, stft_n_mels, mel_f_min,
                audio_augmentations, spectrogram_augmentations, reduce_lr_patience, early_stopping_patience,
                early_stopping_metric, checkpoint_metric, encodec_model, encodec_decode, bandwidth, dataset_type
            ])
        btn.click(fn=train, inputs=inp, outputs=out)

    demo.launch()


if __name__ == '__main__':
    main()
