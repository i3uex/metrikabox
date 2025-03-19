# MetrikaBox

This framework allows you to process an audio dataset for classification tasks using `keras` models. You can configure the processing parameters through command line arguments, which are described below.

## 1. Install

### 1.1 Clone the repository:

```bash
git clone https://github.com/i3uex/metrikabox
cd metrikabox
```

### 1.2 (Optional) Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```

### or using `conda`:

```bash
conda create --name metrikabox python=3.11 --yes
conda activate metrikabox
```

### 1.3 Install the dependencies:

```bash
pip install -r requirements.txt
```

## 2. Usage with Gradio

### 2.1 Training audio classification models

In order to run the audio classification models training interface, run the `demo_train.py` script with the following syntax:

```bash
python3 demo_train.py
```

### 2.2 Prediction with trained audio classification models

In order to run the prediction interface with trained audio classification, run the `demo_predict.py` script with the following syntax:

```bash
python3 demo_predict.py
```

## 3. Usage with command line arguments

### 3.1 Training audio classification models

In order to run the script with command line arguments, use the following syntax:

```bash
python3 main.py train /path/to/dataset
```

#### 3.1.1 Mandatory arguments

- `folder`: Path to the directory containing the audio files to process.

#### 3.1.2 Optional arguments

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
(Available options: Any of https://keras.io/api/optimizers/))

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

- `--model`: Specifies the keras.applications model to use for classification.
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


#### 3.1.3 Advanced usage

```bash 
python3 main.py train /path/to/dataset/ --model_id "my_model" -sr 16000 --window 2 --step 1 --batch_size 32 --epochs 10 --learning_rate 0.001 --audio_augmentations WhiteNoiseAugmentation
```

The command will run the script on the dataset in the data/ folder, specifying a series of parameters for the model, the sample rate, the window size, and other processing configurations.

### 3.2 Prediction with trained audio classification models

In order to run the script with command line arguments, use the following syntax:

```bash
python3 main.py predict /path/to/file /path/to/model  
```

#### 3.2.1 Mandatory arguments

- `file`: Path to the audio file to process.
- `model`: Path to the model to use for prediction.

#### 3.2.2 Optional arguments

- `--model_config_path`: Path to the model configuration file.
- `--task`: Task to perform with the model (classify o segment). Default: segment
