import math
import numpy as np
import torch
import typing as tp
from encodec import binary
from encodec.quantization.ac import ArithmeticDecoder, build_stable_quantized_cdf
from encodec.model import EncodecModel
from audio_classifier.loaders import BaseLoader

MODELS = {
    'encodec_24khz': EncodecModel.encodec_model_24khz,
    'encodec_48khz': EncodecModel.encodec_model_48khz,
}

MODELS2SR = {
    'encodec_24khz': 75,
    'encodec_48khz': 150,
}

class EncodecLoader(BaseLoader):

    def __init__(self, model: str=list(MODELS.keys())[0], decode=False, expected_codebooks=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.expected_codebooks = expected_codebooks if expected_codebooks else (4 if model == 'encodec_48khz' else 8)
        self.window_frames = self.window * MODELS2SR[model]
        self.step_frames = self.step * MODELS2SR[model]
        self.decode = decode

    def decompress_from_file(self, fo: tp.IO[bytes], device='cuda') -> tp.Tuple[torch.Tensor, int, int]:
        """Decompress from a file-object.
        Returns a tuple `(wav, sample_rate)`.

        Args:
            fo (IO[bytes]): file-object from which to read. If you want to decompress
                from `bytes` instead, see `decompress`.
            device: device to use to perform the computations.
        """
        metadata = binary.read_ecdc_header(fo)
        model_name = metadata['m']
        audio_length = metadata['al']
        num_codebooks = metadata['nc']
        use_lm = metadata['lm']
        if model_name != self.model:
            raise ValueError(f"Model mismatch: {model_name} != {self.model}")
        assert isinstance(audio_length, int)
        assert isinstance(num_codebooks, int)
        if num_codebooks != self.expected_codebooks:
            raise ValueError(f"Expected {self.expected_codebooks} codebooks, got {num_codebooks}.")
        if model_name not in MODELS:
            raise ValueError(f"The audio was compressed with an unsupported model {model_name}.")
        model = MODELS[model_name]().to(device)
        if use_lm:
            lm = model.get_lm_model()

        frames: tp.List[np.array] = []
        segment_length = model.segment_length or audio_length
        segment_stride = model.segment_stride or audio_length
        for offset in range(0, audio_length, segment_stride):
            this_segment_length = min(audio_length - offset, segment_length)
            frame_length = int(math.ceil(this_segment_length * model.frame_rate / model.sample_rate))
            if use_lm:
                decoder = ArithmeticDecoder(fo)
                states: tp.Any = None
                offset = 0
                input_ = np.zeros((1, num_codebooks, 1), dtype=np.int64)
            else:
                unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)
            frame = np.zeros((1, num_codebooks, frame_length), dtype=np.int64)
            for t in range(frame_length):
                if use_lm:
                    with torch.no_grad():
                        probas, states, offset = lm(torch.from_numpy(input_), states, offset)
                code_list: tp.List[int] = []
                for k in range(num_codebooks):
                    if use_lm:
                        q_cdf = build_stable_quantized_cdf(
                            probas[0, :, k, 0], decoder.total_range_bits, check=False)
                        code = decoder.pull(q_cdf)
                    else:
                        code = unpacker.pull()
                    if code is None:
                        raise EOFError("The stream ended sooner than expected.")
                    code_list.append(code)
                frame[0, :, t] = np.array(code_list, dtype=np.int64)
                if use_lm:
                    input_ = 1 + frame[:, :, t: t + 1]
            frames.append(frame)
        # Model 48kHz divides into multiple frames
        if len(frames) > 1:
            frames = np.concatenate(frames, axis=-1)
        else:
            frames = np.array(frames[0])
        # Decode RVQ
        if self.decode:
            frames = model.quantizer.decode(torch.from_numpy(frames.transpose(1, 0, 2)).to(device)).detach().cpu().numpy()
        # Or normalize array data
        else:
            frames = frames / model.quantizer.bins
        return frames[0].T, MODELS2SR[model_name], num_codebooks

    def _window(self, a, shape):
        s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
        strides = a.strides + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)

    def load_enc(self, file):
        with open(file, 'rb') as f:
            x, sr, item_len = self.decompress_from_file(f)
        return self._window(x, (self.window_frames, item_len))[::self.step_frames, 0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    encl = EncodecLoader(window=2, step=1)
    w = encl.load_enc("/home/dejavu/repos/AudioClassifierRepo/out.ecdc")
    encl = EncodecLoader(window=2, step=1, decode=True)
    w = encl.load_enc("/home/dejavu/repos/AudioClassifierRepo/out.ecdc")
    encl = EncodecLoader(window=2, step=1, model='encodec_48khz')
    w = encl.load_enc("/home/dejavu/repos/AudioClassifierRepo/samples/concatenated.ecdc")
    encl = EncodecLoader(window=2, step=1, model='encodec_48khz', decode=True)
    w = encl.load_enc("/home/dejavu/repos/AudioClassifierRepo/samples/concatenated.ecdc")

    for i in range(3):
        plt.imshow(w[i], aspect='auto')
        plt.show()

