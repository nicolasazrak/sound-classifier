import tensorflow as tf
import numpy as np
import os.path
from dataset import make_datasets
from keras import *
from kapre import *
from keras.layers import *
from keras.metrics import *
from keras.callbacks import EarlyStopping


def build_lite_rnn_model():
    # Although it would be probably better to input the raw samples directly and
    # calculate the melspectogram inside the neural network, doing that
    # avoids to be able to save this NN in a tensorflow lite format.
    # Using tf lite allowed me to run this code in a raspberry pi
    X_input = tf.keras.Input(shape=(87, 128,))

    X = BatchNormalization()(X_input)
    X = Conv1D(filters=64, kernel_size=2, strides=1)(X)
    X = BatchNormalization()(X)

    X = Activation("relu")(X)
    X = Dropout(rate=0.7)(X)

    X = GRU(units=64, return_sequences=False, bias_regularizer='l2')(X)
    X = Dropout(rate=0.7)(X)
    X = BatchNormalization()(X)

    X = Dense(10, kernel_regularizer='l2')(X)
    X = Dense(1, activation='sigmoid', kernel_regularizer='l2')(X)

    return Model(inputs=X_input, outputs=X)


model = build_lite_rnn_model()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.95,
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[
    Precision(),
    Recall(),
])

train_dataset, test_dataset = make_datasets()
model.summary()
callback = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, mode='min')
model.fit(x=train_dataset, validation_data=test_dataset, epochs=100, callbacks=[callback], class_weight={0: 0.1, 1: 0.9})

print('Evaluating model')
model.evaluate(test_dataset)

# Save the model in lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
