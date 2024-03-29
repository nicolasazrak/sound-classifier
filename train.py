import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
from glob import glob
from math import floor
from random import Random
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.callbacks import EarlyStopping


yamnet_model_handle = 'https://hub.tensorflow.google.cn/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)


def load_wav_file(file_path):
    """ read in a waveform file and convert to 16 kHz mono """
    file_contents = tf.io.read_file(file_path)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)

    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)

    wav = wav[:32000]
    zero_padding = tf.zeros([32000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([wav, zero_padding], 0)

    return wav


def load_numpy_file(file_path):
    return tf.constant(np.load(file_path))


def load_tensor_audio(file_path):
    matches = tf.strings.regex_full_match(file_path, ".*\.wav$")
    return tf.cond(matches, lambda: load_wav_file(file_path), lambda: tf.py_func(load_numpy_file, inp=[file_path], Tout=tf.float32))


def to_embedding(wav_tensor):
    scores, embeddings, spectrogram = yamnet_model(wav_tensor)
    embeddings = tf.reshape(embeddings, (4, 1024))
    return embeddings


def to_label(file_path):
    if "negative" in file_path:
        return tf.constant(0)
    return tf.constant(1)


def make_datasets():
    filenames = []
    filenames.extend(glob("training-data/negative/*.wav"))
    filenames.extend(glob("training-data/positive/*.wav"))
    Random(50).shuffle(filenames)

    # filenames = filenames [:10]
    split_idx = floor(len(filenames) * 0.85)

    train_filenames = filenames[:split_idx]
    train_embeddings = tf.data.Dataset.from_tensor_slices(train_filenames).map(load_tensor_audio).map(to_embedding)
    train_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, train_filenames)))
    train_dataset = tf.data.Dataset.zip((train_embeddings, train_labels)).batch(128).prefetch(tf.data.AUTOTUNE).cache()

    test_filenames = filenames[split_idx:]
    test_embeddings = tf.data.Dataset.from_tensor_slices(test_filenames).map(load_tensor_audio).map(to_embedding)
    test_labels = tf.data.Dataset.from_tensor_slices(list(map(to_label, test_filenames)))
    test_dataset = tf.data.Dataset.zip((test_embeddings, test_labels)).batch(128).prefetch(tf.data.AUTOTUNE).cache()

    return train_dataset, test_dataset


def build_model():
    X_input = tf.keras.Input(shape=(4, 1024,), dtype=tf.float32, name='input_embedding')

    X = X_input
    X = Dense(512, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(X)
    X = Dropout(0.2)(X)
    X = Dense(300, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(X)
    X = Dropout(0.2)(X)
    X = Dense(100, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(X)
    X = Dropout(0.2)(X)
    X = Dense(100, activation='relu', kernel_regularizer='l2', bias_regularizer='l2')(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer='l2', bias_regularizer='l2')(X)

    return Model(inputs=X_input, outputs=X)


def compile_model(model):
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[
        TruePositives(name='tp'),
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        Precision(),
        Recall(),
    ])


def train_model(model):
    train_dataset, test_dataset = make_datasets()
    model.summary()
    callback = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, mode='min')
    model.fit(x=train_dataset, validation_data=test_dataset, epochs=100, callbacks=[callback])
    # model.fit(x=train_dataset, validation_data=test_dataset, epochs=100, callbacks=[callback], class_weight={0: 0.1, 1: 0.9})
    model.evaluate(test_dataset)


def save_model(model):
    # Save the model in lite format.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    # Save the model.
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)


model = build_model()
compile_model(model)
train_model(model)
save_model(model)
