import io
import wave
import time
import pyaudio
import os.path
import threading
import numpy as np
import tflite_runtime.interpreter as tflite
from recorder import Recorder


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


def main():
    print("main()")
    recorder = Recorder(record_seconds=4, samples_format=pyaudio.paFloat32)
    threading.Thread(target=recorder.start).start()
    predictor = Predictor()
    yamnet = Yamnet()
    time.sleep(1)  # Improve this !
    print("Starting!")
    try:
        while True:
            time.sleep(2)  # Yes, this sucks!
            start = time.time()
            predicted = 0

            samples = recorder.get_last_samples()
            samples = list(map(lambda b: np.frombuffer(b), samples))
            samples = [item for sublist in samples for item in sublist]
            if len(samples) > 30000:
                embeddings = yamnet.predict(samples)
                prediction = predictor.predict(embeddings)
                print(prediction)
            else:
                print("skipping prediction")
            # predicted = predictor.predict(loaded_audio)
            # if predicted > 0.5:
            #     print(f"Predicted {predicted}. Took: {time.time() - start} seconds. Saving!")
            #     temp_file_name = os.path.join("training-data", "recognized", "tmp-" + str(time.time()) + ".wav")
            #     with open(temp_file_name, "wb") as outfile:
            #         f.seek(0)
            #         outfile.write(f.getbuffer())

    except:
        print("Stopping")
        recorder.stop()
        raise


main()
