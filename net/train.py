import tensorflow as tf
import numpy as np
import os.path
from model import model
from dataset import make_datasets
from keras import *
from kapre import *
from keras.layers import *
from keras.metrics import *
from log_mel_spectogram import LogMelSpectrogram
from keras.callbacks import ModelCheckpoint
from kapre import STFT, Magnitude, MagnitudeToDecibel


# checkpoint = ModelCheckpoint("model.hdf5", monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto')

train_dataset, test_dataset = make_datasets()

model.summary()

model.fit(x=train_dataset, validation_data=test_dataset, epochs=50, callbacks=[])

# Save the model in lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
