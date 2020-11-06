import tensorflow as tf
import numpy as np
from glob import glob
from math import floor
from random import shuffle
from keras import *
from keras.layers import *
from log_mel_spectogram import LogMelSpectrogram
from keras.callbacks import ModelCheckpoint
from kapre import STFT, Magnitude, MagnitudeToDecibel


def make_datasets():
    all_files = []
    for file_path in glob("training-data/background/*.wav"):
        all_files.append((file_path, 0))
    for file_path in glob("training-data/positive-noisy/*.wav"):
        all_files.append((file_path, 1))
    for file_path in glob("training-data/positive-clean/*.wav"):
        all_files.append((file_path, 1))
    shuffle(all_files)
    idx = floor(len(all_files) * 0.8)
    return all_files[:idx], all_files[idx:]


def generator_for(dataset, batch_size=64):
    current_batch_x = []
    current_batch_y = []

    for audio_path, label in dataset:
        audio = tf.io.read_file(audio_path)
        decoded, sr = tf.audio.decode_wav(
            audio, desired_channels=1, desired_samples=44100)

        current_batch_x.append(decoded.numpy())
        current_batch_y.append(tf.constant([label]))

        if len(current_batch_x) >= batch_size:
            yield (tf.constant(current_batch_x), tf.constant(current_batch_y))
            current_batch_x.clear()
            current_batch_y.clear()

    if len(current_batch_x) > 0:
        yield (tf.constant(current_batch_x), tf.constant(current_batch_y))


def make_pairs(dataset, batch_size=64):
    X = []
    Y = []

    for audio_path, label in dataset:
        audio = tf.io.read_file(audio_path)
        decoded, sample_rate = tf.audio.decode_wav(
            audio, desired_channels=1, desired_samples=44100)
        X.append(decoded.numpy())  # This sucks I know
        Y.append(np.array([label]))

    X = tf.constant(X)
    Y = tf.constant(Y)
    return X, Y


def build_rnn_model(sample_rate, duration, fft_size, hop_size, n_mels):
    n_samples = sample_rate * duration

    X_input = tf.keras.Input(shape=(n_samples,))

    X = LogMelSpectrogram(sample_rate, fft_size, hop_size,
                          n_mels, expand_channels=False)(X_input)

    X = BatchNormalization()(X)

    X = Conv1D(filters=128, kernel_size=3, strides=1)(X)
    X = Conv1D(filters=192, kernel_size=3, strides=1)(X)
    X = BatchNormalization()(X)
    # X = Activation("relu")(X)
    X = Dropout(rate=0.5)(X)

    X = GRU(units=196, return_sequences=True)(X)
    X = Dropout(rate=0.5)(X)
    X = BatchNormalization()(X)

    X = GRU(units=196, return_sequences=False)(X)
    X = Dropout(rate=0.5)(X)
    X = BatchNormalization()(X)

    X = Dense(1)(X)

    model = Model(inputs=X_input, outputs=X)

    return model


model = build_rnn_model(
    sample_rate=22050,
    duration=2,
    fft_size=1024,
    hop_size=256,
    n_mels=512
)

checkpoint = ModelCheckpoint("model.hdf5", monitor='loss',
                             verbose=1, save_weights_only=True, save_best_only=True, mode='auto')

# model.load_weights("model.hdf5")


train, test = make_datasets()
x_train, y_train = make_pairs(train)
x_test, y_test = make_pairs(test)

model.summary()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[
    tf.keras.metrics.BinaryAccuracy(name='70_threshold', threshold=0.7),
    tf.keras.metrics.BinaryAccuracy(name='90_threshold', threshold=0.9),
])

model.fit(x=x_train, y=y_train, validation_data=(
    x_test, y_test), epochs=50, callbacks=[checkpoint])

loss = model.evaluate(x_test,  y_test, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
