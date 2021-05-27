import wave
import numpy as np
import scipy.signal as signal
from real_time_classifier import Yamnet, Predictor


yamnet = Yamnet()
predictor = Predictor()


def eval_wav(file_name):
    with wave.open(file_name, "rb") as f:
        bytes = f.readframes(f.getnframes())
        rate = f.getframerate()
        int_samples = np.frombuffer(bytes, dtype=np.int16)
        s = int_samples.astype(np.float32)
        s = s / (2 ** 15 - 1)
        s = signal.resample(s, 32000)
        embeddings = yamnet.predict(s)
        return predictor.predict(embeddings)


print(eval_wav("training-data/recognized/tmp-1621909207.982633.wav"))
