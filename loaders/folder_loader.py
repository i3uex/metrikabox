import glob
import pickle
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock
import numpy as np
from tqdm import tqdm

from loaders.class_loader import ClassLoader
from config import SAMPLE_RATE, USE_MMAP, CONTEXT_WINDOW, PROCESSING_STEP
from loaders.file_loader import FileLoader

BASE_PATH = ""

class FolderLoader:
    def __init__(self, sample_rate=SAMPLE_RATE, window=CONTEXT_WINDOW, step=PROCESSING_STEP, use_mmap=USE_MMAP, class_loader=ClassLoader(), out_folder=BASE_PATH):
        self.mtx = Lock()
        self.Y = []
        self.X = []
        self.sr = sample_rate
        self.window = window
        self.step = step
        self.use_mmap = use_mmap
        self.class_loader = class_loader
        self.file_loader = FileLoader(sample_rate=self.sr, window=self.window, step=self.step)
        self.MMAP_PATH = out_folder + "MMAP.npy"
        self.CLASSES_PATH = out_folder + "CLASSES.npy"
        self.MMAP_SHAPE_FILE = out_folder + "MMAP_shape.pkl"

    def load(self, folder, max_files=None):
        items = glob.glob(folder + "*/*", recursive=True)
        if max_files:
            items = items[:max_files]
        # Load audio files
        with ProcessPoolExecutor() as ex:
            futures = [ex.submit(self.file_loader.load_from_file, audio_file) for audio_file in items]
            with tqdm(total=len(futures)) as pbar:
                # Process loaded audios
                for af, future in zip(items, futures):
                    try:
                        x = future.result()
                    except Exception as e:
                        print(f"Exception {type(e)} in file {items}. Skipping")
                    if self.use_mmap:
                        try:
                            with open(self.MMAP_SHAPE_FILE, "rb") as f:
                                out_shape = pickle.load(f)
                            offset = out_shape[0] * out_shape[1] * 4
                            mmap = np.memmap(self.MMAP_PATH, dtype=x.dtype, shape=x.shape, mode="r+", offset=offset)
                        except FileNotFoundError:
                            out_shape = [0, SAMPLE_RATE*2]
                            mmap = np.memmap(self.MMAP_PATH, dtype=x.dtype, shape=x.shape, mode="w+")
                        # Writting into memmap
                        mmap[:] = x[:]
                        mmap.flush()
                        out_shape[0] += mmap.shape[0]
                        with open(self.MMAP_SHAPE_FILE, "wb") as f:
                            pickle.dump(out_shape, f)
                    else:
                        self.X.extend(x)
                    self.Y.extend(self.class_loader.get_class(af, x.shape[0]))
                    pbar.update()
        seed = 42
        rstate = np.random.RandomState(seed)
        rstate.shuffle(self.Y)
        if not self.use_mmap:
            self.X = np.array(self.X)
            rstate = np.random.RandomState(seed)
            rstate.shuffle(self.X)
        else:
            X = np.memmap(self.MMAP_PATH, dtype=np.float32, mode="r+")
            X = X.reshape(X.size//(SAMPLE_RATE*self.window), SAMPLE_RATE*self.window, 1)
            rstate = np.random.RandomState(seed)
            rstate.shuffle(X)
            X.flush()
            self.X = X
            with open(self.CLASSES_PATH, "wb") as f:
                pickle.dump(self.Y, f)
        return self.X, self.Y
