import librosa
import numpy as np
import tensorflow as tf
from glob import glob
from math import floor
from random import Random
from utils import load_audio


def load_tensor_audio(file_or_path):
    return tf.constant(load_audio(file_or_path))


def to_label(file_path):
    if "background" in file_path:
        return 0
    return 1


def make_datasets():
    filenames = []
    filenames.extend(glob("training-data/background/*.wav"))
    filenames.extend(glob("training-data/positive-noisy/*.wav"))
    filenames.extend(glob("training-data/positive-clean/*.wav"))
    Random(5).shuffle(filenames)

    split_idx = floor(len(filenames) * 0.8)

    train_filenames = filenames[:split_idx]
    train_audios = tf.data.Dataset.from_tensor_slices(list(map(load_tensor_audio, train_filenames)))
    train_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, train_filenames)))
    train_dataset = tf.data.Dataset.zip((train_audios, train_labels)).batch(64)

    test_filenames = filenames[split_idx:]
    test_audios = tf.data.Dataset.from_tensor_slices(list(map(load_tensor_audio, test_filenames)))
    test_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, test_filenames)))
    test_dataset = tf.data.Dataset.zip((test_audios, test_labels)).batch(64)

    return train_dataset, test_dataset
