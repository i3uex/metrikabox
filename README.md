# MetrikaBox

MetrikaBox is an open framework for experimenting with audio classification. 

This framework allows you to process an audio dataset for classification tasks using [Keras][keras] models. You can configure the processing parameters through command line arguments, which are described below.

[keras]: https://keras.io/ "The purpose of Keras is to give an unfair advantage to any developer looking to ship Machine Learning-powered apps"

## Table of Contents

1. [Installation Process](#1-installation-process)
2. [Datasets and Samples](#2-datasets-and-samples)
3. [Usage with Gradio](#3-usage-with-gradio)
4. [Usage with command line arguments](#4-usage-with-command-line-arguments)

## 1. Installation Process

Follow these steps to get your computer ready to use this project:

1. Clone the repository:

    ```shell
    git clone https://github.com/i3uex/metrikabox
    cd metrikabox
    ```

2. Optionally, create and activate a virtual environment, either with Python:

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

    > Note: If on Windows, change your path to **venv\Scripts\activate**.

    or with Conda:

    ```shell
    conda create --name metrikabox python=3.11 --yes
    conda activate metrikabox
    ```

3. Install the dependencies:

    ```shell
    pip install -r requirements.txt
    ```

4. If using Miniconda, you can avoid stems 2 and 3, and recreate the Conda virtual environment from scratch with the following lines:

    ```shell
    conda env create --file project_environment.yml
    conda activate metrikabox
    ```

5. Install [FFmpeg][ffmpeg] with the following command in Linux:

    ```bash
    sudo apt install ffmpeg
    ```

    If using another operating system, refer to FFmpeg page for the installation instructions.

    [ffmpeg]: https://www.ffmpeg.org/ "A complete, cross-platform solution to record, convert and stream audio and video"

> **Note:** If you no longer need the Conda environment, just deactivate it with `conda deactivate` and delete it with `conda remove --name metrikabox --all --yes`.

## 2. Datasets and Samples

You can download two example datasets ([GTZAN Speech & Music][gtzan_musicspeech_collection] and [GTZAN Genres][gtzan]) by running the script `bash download_datasets.sh`.

[gtzan_musicspeech_collection]: https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection "GTZAN music/speech collection"
[gtzan]: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification "GTZAN Dataset - Music Genre Classification"

You can download four example audio files (taken from [librosa's data][librosa_data] repository) by running the script `bash download_samples.sh`.

[librosa_data]: https://github.com/librosa/data "Example (audio) data for use with librosa."

## 3. Usage with Gradio

### 3.1. Training Audio Classification Models

In order to run the audio classification models training interface, run the `demo_train.py` script with the following syntax:

```bash
python demo_train.py
```

### 3.2. Prediction with Trained Audio Classification Models

In order to run the prediction interface with trained audio classification, run the `demo_predict.py` script with the following syntax:

```bash
python demo_predict.py
```

## 4. Usage with Command Line Arguments

### 4.1. Training Audio Classification Models

In order to run the script with command line arguments, use the following syntax:

```bash
python main.py train "datasets/GTZAN Speech_Music"
```

This will train a model using the GTZAN Speech&Music classification dataset, that will be capable of classifying audio files into two classes: "Speech" and "Music".

#### 4.1.1. Mandatory Arguments

- `--folder`: Path to the directory containing the folders with the class labels. Each folder must contain the audio files of a single class. 

#### 4.1.2. Optional Arguments

- `--sample_rate`: Sample rate to which the audios will be converted. This value must be an integer.
(Default: 22050)

- `--window`: Time in seconds to be processed as a single element. This value must be a decimal number.
(Default: 2)

- `--step`: Time in seconds to jump between windows. It is recommended that it be at most half of - `window to ensure overlap and not lose information.
(Default: 1)

- `--classes2avoid`: List of classes to avoid in the dataset (these classes will not be loaded).

- `--checkpoints_folder`: Path to the directory where the model checkpoints will be saved.

- `--optimizer`: Optimizer from `keras.optimizers` to use for model training.
(Default: Adam)
(Available options: Any of https://keras.io/api/optimizers/)

- `--class_loader`: Loader class to use.
(Default: Load classes from the folder name with the audio files)
(Available options: Load classes from the folder name with the audio files)

- `--learning_rate`: Learning rate for the optimizer. This value must be a decimal number.
(Default: 0.001)

- `--model_id`: ID of the model to use 
(Default: A combination of the Current time, the configured sample rate and the processing window and step separated by _).

- `--stft_nfft`: Length of the FFT window. This value must be an integer.
(Default: 1024)

- `--stft_win`: Length of the window for each audio frame before padding to match stft_nfft.
(Default: 1024)

- `--stft_hop`: Number of samples between successive frames.
(Default: 256)

- `--stft_nmels`: Number of Mel bands to generate.
(Default: 128)

- `--mel_f_min`: Lowest frequency of the Mel filter band.
(Default: 0)

- `--model`: Specifies the `keras.applications` model to use for classification.
(Default: MNIST (https://keras.io/examples/vision/mnist_convnet/))
(Available options: Any of https://keras.io/api/applications/)

- `--audio_augmentations`: List of audio augmentations to apply. If none are specified, no augmentations will be used.
(Default: None)
(Available options: `WhiteNoiseAugmentation`)

- `--spectrogram_augmentations`: List of spectrogram augmentations to apply. If none are specified, no augmentations will be used.
(Default: None)
(Available options: `SpecAugmentation`)

- `--batch_size`: Size of the batch for model training.

- `--epochs`: Number of epochs to train the model.

#### 4.1.3. Advanced Usage

In this case we will train a classification model using the GTZAN Genres classification dataset running the following command:

```bash 
python main.py train "datasets/GTZAN Genre/Data/genres_original" --model keras.MobileNetV2 --model_id "GTZAN_Genres" -sr 16000 --window 5 --step 2.5 --batch_size 32 --epochs 100 --learning_rate 0.001 --audio_augmentations [WhiteNoiseAugmentation]
```

This model will be capable of classifying audio files into 10 different music genres. In this case we use the folder containing the 10 subfolders, each one containing audio files of a different genre.

The command will run the script on the dataset in the genres_original/ folder, specifying a series of parameters for the model, the sample rate, the window size, and other processing configurations.

### 4.2. Prediction with Trained Audio Classification Models

In order to run the script with command line arguments, use the following syntax:

```bash
python main.py predict samples/audio2.ogg checkpoints/GTZAN_Genres.keras checkpoints/model_config/GTZAN_Genres/model-config.json
```

#### 4.2.1. Mandatory Arguments

- `--file`: Path to the audio file to process.
- `--model`: Path to the model to use for prediction.

#### 4.2.2 Optional Arguments

- `--model_config_path`: Path to the model configuration file.
- `--task`: Task to perform with the model (classify or segment). Default: segment
