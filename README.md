# Audio Processing Script

Este script permite procesar un conjunto de datos de audio para tareas de clasificación utilizando modelos de `keras`. Puedes configurar los parámetros de procesamiento a través de argumentos de línea de comandos, los cuales se describen a continuación.


## Install

Clona el repositorio:

```bash
git clone https://github.com/usuario/nombre_repositorio.git
cd nombre_repositorio
```
(Opcional) Crea y activa un entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows usa: venv\Scripts\activate
```
Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

```bash
python script.py folder [options]
```

## Argumentos
### Argumentos Posicionales

- `folder`: Ruta al directorio que contiene los archivos de audio a procesar.

### Argumentos Opcionales

- `model_id`: ID del modelo a utilizar (Default: A combination of the Current time, the configured sample rate and the processing window and step separated by _).

- `sample_rate`: Tasa de muestreo a la que se convertirán los audios. Este valor debe ser un número entero.
(Default: 22050)

- `window`: Tiempo en segundos que se procesará como un solo elemento. Este valor debe ser un número decimal.
(Default: 2)

- `step`: Tiempo en segundos para saltar entre ventanas. Se recomienda que sea como máximo la mitad de - `window para asegurar solapamiento y no perder información.
(Default: 1)

- `use_mmap`:Indica si se debe crear un archivo intermedio en formato mmap para almacenar los datos durante el procesamiento, útil cuando no hay suficiente memoria RAM disponible.
(Default: False)

- `batch_size`: Tamaño del lote para el entrenamiento. Este valor debe ser un número entero.
(Default: 128)

- `epochs`: Número de épocas para entrenar el modelo. Este valor debe ser un número entero.
(Default: 150)

- `stft_nfft`: Longitud de la ventana FFT. Este valor debe ser un número entero.
(Default: 1024)

- `stft_win`: Longitud de la ventana para cada cuadro de audio antes de aplicar el relleno para coincidir con stft_nfft.
(Default: 1024)

- `stft_hop`: Número de muestras entre cuadros sucesivos.
(Default: 256)

- `stft_nmels`: Número de bandas Mel a generar.
(Default: 128)

- `mel_f_min`: Frecuencia más baja de la banda de filtro Mel.
(Default: 0)

- `model`: Especifica el modelo de keras.applications a utilizar para la clasificación.
(Default: MNIST (https://keras.io/examples/vision/mnist_convnet/))
(Available options: Any of https://keras.io/api/applications/)

- `optimizer`: Especifica el optimizador de keras.optimizers a utilizar para el entrenamiento del modelo.
(Default: Adam)
(Available options: Any of https://keras.io/api/optimizers/))

- `class_loader`: Clase de cargador de clases a utilizar.
(Default: Cargar clases desde el nombre de la carpeta con los archivos de audio)
(Available options: Cargar clases desde el nombre de la carpeta con los archivos de audio)

- `learning_rate`: Tasa de aprendizaje para el optimizador. Este valor debe ser un número decimal.
(Default: 0.001)

- `trainset_shuffle_size`: Tamaño del buffer de mezcla para el conjunto de datos de entrenamiento. Este valor debe ser un número entero.
(Default: 1024)

- `audio_augmentations`: Lista de aumentaciones de audio a aplicar. Si no se especifica ninguna, no se utilizarán aumentaciones.
(Default: None)
(Available options: `WhiteNoiseAugmentation`)

- `spectrogram_augmentations`: Lista de aumentaciones de espectrogramas a aplicar. Si no se especifica ninguna, no se utilizarán aumentaciones.
(Default: None)
(Available options: `SpecAugmentation`)

- `classes2avoid`: Lista de clases a evitar en el conjunto de datos (estas clases no serán cargadas).


## Ejemplo avanzado de uso

```bash 
python script.py data/ --model_id "my_model" -sr 16000 --window 2 --step 1 --batch_size 32 --epochs 10 --learning_rate 0.001 --audio_augmentations WhiteNoiseAugmentation
```
Este comando ejecutará el script sobre el conjunto de datos en la carpeta data/, especificando una serie de parámetros para el modelo, la tasa de muestreo, el tamaño de ventana y otras configuraciones de procesamiento.
