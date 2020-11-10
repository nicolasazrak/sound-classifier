import tensorflow as tf
import numpy as np
import os.path
from dataset import make_datasets
from keras import *
from kapre import *
from keras.layers import *
from keras.metrics import *
from keras.callbacks import ModelCheckpoint


def build_lite_rnn_model():
    X_input = tf.keras.Input(shape=(87, 128,))

    X = BatchNormalization()(X_input)
    X = Conv1D(filters=64, kernel_size=8, strides=4)(X)
    X = BatchNormalization()(X)

    X = Activation("relu")(X)
    X = Dropout(rate=0.7)(X)

    X = GRU(units=128, return_sequences=False)(X)
    X = Dropout(rate=0.7)(X)
    X = BatchNormalization()(X)

    X = Dense(10, kernel_regularizer='l2')(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer='l2')(X)

    return Model(inputs=X_input, outputs=X)


model = build_lite_rnn_model()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=200,
    decay_rate=0.95,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[
    'binary_accuracy',
    Precision(),
    Recall(),
])

train_dataset, test_dataset = make_datasets()

model.summary()
model.fit(x=train_dataset, validation_data=test_dataset, epochs=100, callbacks=[])

# Save the model in lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
