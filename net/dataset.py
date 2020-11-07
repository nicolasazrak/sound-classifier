from glob import glob
from math import floor
import tensorflow as tf
from random import Random


def load_audio(file_path):
    binary = tf.io.read_file(file_path)
    decoded, _ = tf.audio.decode_wav(binary, desired_channels=1, desired_samples=44100)
    return decoded


def to_label(file_path):
    if "background" in file_path:
        return tf.constant(0)
    return tf.constant(1)


def make_datasets():
    filenames = glob("training-data/*/*.wav")
    Random(1).shuffle(filenames)

    split_idx = floor(len(filenames) * 0.8)

    train_filenames = filenames[:split_idx]
    train_audios = tf.data.Dataset.from_tensor_slices(train_filenames).map(load_audio)
    train_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, train_filenames)))
    train_dataset = tf.data.Dataset.zip((train_audios, train_labels)).batch(32)

    test_filenames = filenames[split_idx:]
    test_audios = tf.data.Dataset.from_tensor_slices(test_filenames).map(load_audio)
    test_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, test_filenames)))
    test_dataset = tf.data.Dataset.zip((test_audios, test_labels)).batch(32)

    return train_dataset, test_dataset
