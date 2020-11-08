from glob import glob
from math import floor
import librosa
import numpy as np
import tensorflow as tf
from random import Random


def load_audio(file_path):
    padded = np.zeros((44100,))
    y, sr = librosa.load(file_path, sr=22050, duration=2)
    padded[:y.shape[0]] = y[:]
    spect = librosa.feature.melspectrogram(y=padded, sr=sr)
    swapped = np.swapaxes(spect, 0, 1)
    return tf.constant(swapped)


def to_label(file_path):
    if "background" in file_path:
        return 0
    return 1


def make_datasets():
    filenames = []
    filenames.extend(glob("training-data/background/*.wav"))
    filenames.extend(glob("training-data/positive-noisy/*.wav"))
    filenames.extend(glob("training-data/positive-clean/*.wav"))
    Random(1).shuffle(filenames)

    split_idx = floor(len(filenames) * 0.8)

    train_filenames = filenames[:split_idx]
    train_audios = tf.data.Dataset.from_tensor_slices(list(map(load_audio, train_filenames)))
    # .map(lambda f: tf.py_function(load_audio, [f], [tf.float64]))
    train_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, train_filenames)))
    train_dataset = tf.data.Dataset.zip((train_audios, train_labels)).batch(32)

    test_filenames = filenames[split_idx:]
    test_audios = tf.data.Dataset.from_tensor_slices(list(map(load_audio, test_filenames)))
    # .map(lambda f: tf.py_function(load_audio, [f], [tf.string]))
    test_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, test_filenames)))
    test_dataset = tf.data.Dataset.zip((test_audios, test_labels)).batch(32)

    return train_dataset, test_dataset
