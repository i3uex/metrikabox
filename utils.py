import math
from typing import Union

from pydub import AudioSegment
from pydub.effects import normalize
from librosa.util import buf_to_float
import numpy as np
import random

from config import SAMPLE_RATE, CONTEXT_WINDOW, PROCESSING_STEP

class Singleton(type):
    """
    Metaclass that allows the creation of Singleton classes
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in Singleton._instances:
            Singleton._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return Singleton._instances[cls]

    def clear(cls):
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass

random.seed(42)

MAX_ITEMS = 10000
INPUT_SHAPE = (SAMPLE_RATE*CONTEXT_WINDOW, 1)

def _window(a, window=CONTEXT_WINDOW, step=PROCESSING_STEP, sr=SAMPLE_RATE) -> np.ndarray:
    w = int(window * sr)
    o = int(step * sr)
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    return np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]

def detect_leading_silence(sound: AudioSegment, silence_threshold:float=-25.0, chunk_size:int=10) -> float:
    """
    Iterate over chunks until you find the first one with sound
    Args:
        sound: pydub.AudioSegment to find silence of
        silence_threshold: threshold in dBs to identify as silence
        chunk_size: miliseconds used for every chunk
    Returns:
        time in miliseconds where the silence ends
    """
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def delete_start_and_end_silence(sound: AudioSegment) -> AudioSegment:
    """
    Gets the miliseconds of silence at the start and end and then trims the audio to delete this silences
    Args:
        sound: pydub.AudioSegment to delete silence of
    Returns:
        pydub.AudioSegment with no silence at the start nor the end
    """
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    return sound[start_trim:duration-end_trim]

def load_audio(audio_file):
    try:
        return apply_window(normalize(delete_start_and_end_silence(AudioSegment.from_file(audio_file).set_frame_rate(SAMPLE_RATE).set_channels(1))))
    except Exception as e:
        print(audio_file)
        raise e

def create_speech_music_file(music_file, speech_file, index=None):
  try:
      music = load_audio(music_file)
      speech = load_audio(speech_file)[:music.duration_seconds*1000]
  except Exception as e:
      print(music_file, speech_file)
      raise e
  music = music[:speech.duration_seconds*1000]
  combined = (music - random.uniform(6, 25)).overlay(speech)
  combined.export("speech_music/%d.flac" % index)

def apply_window(audio: Union[AudioSegment, np.array], window=CONTEXT_WINDOW, step=PROCESSING_STEP, sr=SAMPLE_RATE):
    if type(audio) is AudioSegment:
        audio = buf_to_float(audio.get_array_of_samples(), n_bytes=audio.sample_width)

    return np.expand_dims(_window(np.pad(audio, math.ceil(sr/2*window), mode="symmetric"), window=window, step=step, sr=sr), 2)

def apply_window_librosa(audio: np.array, sr=SAMPLE_RATE):
    return np.expand_dims(_window(np.pad(audio, sr), sr=sr), 2)

def prepare_audio(audio, window=CONTEXT_WINDOW, step=PROCESSING_STEP, sr=SAMPLE_RATE):
    audio = AudioSegment.from_file(audio).set_frame_rate(sr).set_channels(1)
    return apply_window(audio, window=window, step=step, sr=sr)

def get_mels_from_hop_and_win_lengths(hop_length, win_length, input_size=SAMPLE_RATE*CONTEXT_WINDOW):
    return int(math.floor((input_size - win_length) / hop_length) + 1)

