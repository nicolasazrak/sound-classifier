import io
import wave
import time
import pyaudio
import datetime
import os.path
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
from recorder import ChunkedRecorder


class Predictor:

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']

    def predict(self, embeddings):
        embeddings = np.expand_dims(embeddings, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], embeddings)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]


class Yamnet:

    def __init__(self):
        self.interpreter = tflite.Interpreter('yamnet.tflite')

        self.input_details = self.interpreter.get_input_details()
        self.waveform_input_index = self.input_details[0]['index']
        self.output_details = self.interpreter.get_output_details()
        self.scores_output_index = self.output_details[0]['index']
        self.embeddings_output_index = self.output_details[1]['index']
        self.spectrogram_output_index = self.output_details[2]['index']

    def predict(self, samples):
        self.interpreter.resize_tensor_input(self.waveform_input_index, [len(samples)], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, np.array(samples, dtype=np.float32))
        self.interpreter.invoke()
        scores, embeddings, spectrogram = (
            self.interpreter.get_tensor(self.scores_output_index),
            self.interpreter.get_tensor(self.embeddings_output_index),
            self.interpreter.get_tensor(self.spectrogram_output_index)
        )
        # print(scores.shape, embeddings.shape, spectrogram.shape)  # (N, 521) (N, 1024) (M, 64)
        return embeddings


class RealTimeClassifier:

    def __init__(self):
        self.predictor = Predictor()
        self.yamnet = Yamnet()
        self.samples_buffer = []

    def analyze(self, recording):
        start = datetime.datetime.now()
        embeddings = self.yamnet.predict(recording.samples_at(16000))
        prediction = self.predictor.predict(embeddings)
        end = datetime.datetime.now()
        print("Prediction: " + str(prediction) + ", took: " + str(end-start))
        if prediction > 0.5:
            file_name = os.path.join("training-data", "recognized", "tmp-" + str(time.time()) + ".wav")
            recording.save_to(file_name)

    def run(self):
        print("Starting!")
        recorder = ChunkedRecorder(recording_duration=2, callback=self.analyze)
        recorder.run()


def main():
    classifier = RealTimeClassifier()
    classifier.run()


main()
