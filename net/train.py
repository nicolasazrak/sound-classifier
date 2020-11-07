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


checkpoint = ModelCheckpoint("model.hdf5", monitor='loss', verbose=1, save_weights_only=True, save_best_only=True, mode='auto')

train_dataset, test_dataset = make_datasets()

model.summary()

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['binary_accuracy'])

model.fit(x=train_dataset, validation_data=test_dataset, epochs=10, callbacks=[checkpoint])

#loss = model.evaluate(x_test,  y_test, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
