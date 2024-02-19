import glob
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Tuple

import numpy as np
from tqdm import tqdm

from loaders.class_loader import ClassLoader
from config import DEFAULT_SAMPLE_RATE, DEFAULT_WINDOW, DEFAULT_STEP
from loaders.file_loader import FileLoader

BASE_PATH = ''

class FolderLoader:
    def __init__(self,
                 sample_rate:int=DEFAULT_SAMPLE_RATE,
                 window:float=DEFAULT_WINDOW,
                 step:float=DEFAULT_STEP,
                 use_mmap:bool=False,
                 class_loader:ClassLoader=ClassLoader(),
                 out_folder:str=BASE_PATH
                 ):
        self.Y = []
        self.X = []
        self.sr = sample_rate
        self.window = window
        self.step = step
        self.use_mmap = use_mmap
        self.class_loader = class_loader
        self.file_loader = FileLoader(sample_rate=self.sr, window=self.window, step=self.step)
        self.MMAP_PATH = f'{out_folder}MMAP.npy'
        self.CLASSES_PATH = f'{out_folder}CLASSES.npy'
        self.MMAP_SHAPE_FILE = f'{out_folder}MMAP_shape.pkl'

    def load(self, folder:str, max_files:int=None, classes2avoid=(), audio_formats=(".wav", ".mp3")) -> Tuple[np.ndarray, list]:
        items = list(filter(lambda x: not os.path.isdir(x) and (any([x.lower().endswith(f) for f in audio_formats]) if audio_formats else True), glob.glob(folder + '**', recursive=True)))
        if max_files:
            items = items[:max_files]
        # Load audio files
        try:
            with open(self.MMAP_SHAPE_FILE, 'rb') as f:
                out_shape = pickle.load(f)
            with open(self.CLASSES_PATH, 'rb') as f:
                Y = pickle.load(f)
            print(out_shape)
            out_shape[1] = int(out_shape[1])
            X = np.memmap(self.MMAP_PATH, dtype=np.float32, mode='r+', shape=tuple(out_shape))
            return X, Y
        except FileNotFoundError as e:
            print(e)
            out_shape = [0, self.sr * self.window]
        num_processes = multiprocessing.cpu_count() // 8 if self.use_mmap else multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=num_processes, mp_context=multiprocessing.get_context("fork")) as ex:
            futures = [ex.submit(self.file_loader.load, audio_file) for audio_file in items]
            with tqdm(total=len(futures), desc='Loading files') as pbar:
                # Process loaded audios
                for i, (af, future) in enumerate(zip(items, futures)):
                    try:
                        x = future.result()
                    except Exception as e:
                        print(f'Exception {type(e)} in file {items[i]}. Skipping')
                    y = self.class_loader.get_class(af, x.shape[0])
                    x, y = list(zip(*filter(lambda _item: _item[1] not in classes2avoid, zip(x, y))))
                    if self.use_mmap:
                        x = np.array(x)
                        try:
                            offset = int(out_shape[0] * out_shape[1] * x.itemsize)
                            mmap = np.memmap(self.MMAP_PATH, dtype=x.dtype, shape=x.shape, mode='r+', offset=offset)
                        except FileNotFoundError:
                            mmap = np.memmap(self.MMAP_PATH, dtype=x.dtype, shape=x.shape, mode='w+')
                        # Writting into memmap
                        mmap[:] = x[:]
                        mmap.flush()
                        out_shape[0] += mmap.shape[0]
                    else:
                        self.X.extend(x)
                    self.Y.extend(y)
                    pbar.update()
        print("Shuffleling dataset") 
        seed = 42
        rstate = np.random.RandomState(seed)
        rstate.shuffle(self.Y)
        if not self.use_mmap:
            self.X = np.array(self.X)
            rstate = np.random.RandomState(seed)
            rstate.shuffle(self.X)
        else:
            with open(self.MMAP_SHAPE_FILE, 'wb') as f:
                pickle.dump(out_shape, f)
            X = np.memmap(self.MMAP_PATH, dtype=np.float32, mode='r+')
            X = X.reshape(X.size//int(self.sr*self.window), int(self.sr*self.window), 1)
            rstate = np.random.RandomState(seed)
            rstate.shuffle(X)
            X.flush()
            self.X = X
            with open(self.CLASSES_PATH, 'wb') as f:
                pickle.dump(self.Y, f)
        return self.X, self.Y
