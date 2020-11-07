import matplotlib.pyplot as plt
import pyaudio
import wave
import threading
import sys
import time
import os.path
from flask import Flask, render_template


class Recorder:

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.should_stop = False
        self.buffer_lock = threading.Lock()
        self.stop_lock = threading.Lock()

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 30

    def save_buffer(self):
        output_name = os.path.join(
            "training-data", "raw", "output-" + str(time.time()) + ".wav")
        wf = wave.open(output_name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        self.buffer_lock.acquire()
        frames_copy = []
        try:
            frames_copy = self.frames.copy()
        finally:
            self.buffer_lock.release()
        wf.writeframes(b''.join(frames_copy))
        wf.close()

        # audio, sr = librosa.core.load(output_name, sr=self.RATE)
        # ax = plt.plot()
        # librosa.display.waveplot(audio, sr=sr, ax=ax)
        # ax.set(title='Monophonic')
        # ax.label_outer()
        # plt.savefig(output_name.replace("wav", "png"))

        return output_name

    def start(self):
        self.stop_lock.acquire()
        stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK
        )

        buffer_frame_size = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)

        while not self.should_stop:
            data = stream.read(self.CHUNK)
            self.buffer_lock.acquire()
            try:
                self.frames.append(data)
                self.frames = self.frames[-buffer_frame_size:]
            finally:
                self.buffer_lock.release()

        stream.stop_stream()
        stream.close()
        self.p.terminate()
        self.stop_lock.release()

    def stop(self):
        self.should_stop = True
        self.stop_lock.acquire()


if __name__ == "__main__":
    recorder = Recorder()
    print("Starting")
    threading.Thread(target=recorder.start).start()

    try:
        for line in sys.stdin:
            print("Saving recording")
            recorder.save_buffer()
    finally:
        print("Stopping")
        recorder.stop()


app = Flask(__name__)

recorder = Recorder()
threading.Thread(target=recorder.start).start()


@app.route('/')
def index():
    return render_template('recorder.html')


@app.route('/save', methods=["POST"])
def save():
    return recorder.save_buffer()


try:
    app.run(debug=True, host='0.0.0.0')
finally:
    recorder.stop()
