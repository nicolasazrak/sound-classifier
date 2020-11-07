import tensorflow as tf
import numpy as np
import os.path
from keras import *
from kapre import *
from keras.layers import *
from keras.metrics import *
from keras.callbacks import ModelCheckpoint
from kapre import STFT, Magnitude, MagnitudeToDecibel


def build_rnn_model(sample_rate, duration, fft_size, hop_size, n_mels):
    n_samples = sample_rate * duration

    X_input = tf.keras.Input(shape=(n_samples,))

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

    # X = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels, expand_channels=False)(X_input)
    X = BatchNormalization()(X)

    X = Conv1D(filters=128, kernel_size=3, strides=1)(X)
    X = MaxPooling1D()(X)
    X = BatchNormalization()(X)

    X = Activation("relu")(X)
    X = Dropout(rate=0.4)(X)

    X = GRU(units=256, return_sequences=True)(X)
    X = Dropout(rate=0.4)(X)
    X = BatchNormalization()(X)

    X = GRU(units=256, return_sequences=False)(X)
    X = Dropout(rate=0.4)(X)
    X = BatchNormalization()(X)

    X = Dense(10)(X)
    X = Dense(1, activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X)

    return model


model = build_rnn_model(
    sample_rate=22050,
    duration=2,
    fft_size=1024,
    hop_size=256,
    n_mels=512
)

model.load_weights("model.hdf5")
