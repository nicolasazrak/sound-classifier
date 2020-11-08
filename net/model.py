import tensorflow as tf
import numpy as np
import os.path
from keras import *
from kapre import *
from keras.layers import *
from keras.metrics import *
from keras.callbacks import ModelCheckpoint
from net.log_mel_spectogram import LogMelSpectrogram
from kapre import STFT, Magnitude, MagnitudeToDecibel


def build_complex_rnn_model(use_kapre, sample_rate, n_mels, fft_size, hop_size):
    X_input = tf.keras.Input(shape=(44100,))
    X = None

    if use_kapre:
        X = Lambda(lambda x: tf.expand_dims(x, axis=-1))(X_input)
        X = STFT(
            n_fft=n_mels,
            win_length=fft_size,
            hop_length=hop_size,
            window_name=None,
            pad_end=False,
            input_data_format='channels_last',
            output_data_format='channels_last'
        )(X)
        X = Magnitude()(X)
        X = MagnitudeToDecibel()(X)
        X = Lambda(lambda x: tf.squeeze(x, axis=-1))(X)
    else:
        X = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels, expand_channels=False)(X_input)

    X = BatchNormalization()(X)

    X = Conv1D(filters=64, kernel_size=8, strides=4)(X)
    X = BatchNormalization()(X)

    X = Activation("relu")(X)
    X = Dropout(rate=0.7)(X)

    X = GRU(units=128, return_sequences=False)(X)
    X = Dropout(rate=0.7)(X)
    X = BatchNormalization()(X)

    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


def build_lite_rnn_model():
    X_input = tf.keras.Input(shape=(87, 128,))

    X = Conv1D(filters=64, kernel_size=8, strides=4)(X_input)
    X = BatchNormalization()(X)

    X = Activation("relu")(X)
    X = Dropout(rate=0.7)(X)

    X = GRU(units=128, return_sequences=False)(X)
    X = Dropout(rate=0.7)(X)
    X = BatchNormalization()(X)

    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


model = build_lite_rnn_model()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[
    'binary_accuracy',
    TruePositives(),
    TrueNegatives(),
    FalsePositives(),
    FalseNegatives(),
])
