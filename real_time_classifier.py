import io
import pyaudio
import wave
import threading
import time
import numpy as np
import librosa
import os.path
from collections import deque
from utils import load_audio
import tflite_runtime.interpreter as tflite


class Predictor:

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="model.tflite")
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']

    def predict(self, samples):
        samples = np.expand_dims(samples, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], samples)

        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        return self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]


class Recorder:

    def __init__(self, queue):
        self.p = pyaudio.PyAudio()
        self.should_stop = False
        self.queue = queue
        self.stop_lock = threading.Lock()

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.RECORD_SECONDS = 2

    def start(self):
        self.stop_lock.acquire()
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        while not self.should_stop:
            frames = []

            for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
                data = stream.read(self.CHUNK)
                frames.append(data)

            self.queue.append(frames)

        stream.stop_stream()
        stream.close()
        self.p.terminate()
        self.stop_lock.release()

    def stop(self):
        self.should_stop = True
        self.stop_lock.acquire()


def pop_audio(q, recorder):
    frames = q.pop()
    f = io.BytesIO()
    wf = wave.open(f, 'wb')
    wf.setnchannels(recorder.CHANNELS)
    wf.setsampwidth(recorder.p.get_sample_size(recorder.FORMAT))
    wf.setframerate(recorder.RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    f.seek(0)
    return f


def main():
    print("main()")
    q = deque([], 1)
    recorder = Recorder(q)
    threading.Thread(target=recorder.start).start()
    predictor = Predictor()

    time.sleep(1)
    print("Starting!")
    try:
        while True:
            time.sleep(2)
            start = time.time()
            predicted = 0
            f = pop_audio(q, recorder)
            loaded_audio = load_audio(f)
            predicted = predictor.predict(loaded_audio)
            if predicted > 0.5:
                print(f"Predicted {predicted}. Took: {time.time() - start} seconds. Saving!")
                temp_file_name = os.path.join("training-data", "recognized", "tmp-" + str(time.time()) + ".wav")
                with open(temp_file_name, "wb") as outfile:
                    f.seek(0)
                    outfile.write(f.getbuffer())

    except:
        print("Stopping")
        recorder.stop()
        raise


main()
