import pyaudio
import wave
import threading
import sys
import time
import numpy as np
from collections import deque
import multiprocessing as mp
import tempfile
import os.path
import tensorflow as tf
from net.model import model
from flask import Flask, render_template


def get_all_queue_result(queue):
    result_list = []
    while not queue.empty():
        result_list.append(queue.get())
    return result_list


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


def main():
    q = deque([], 1)
    recorder = Recorder(q)
    threading.Thread(target=recorder.start).start()

    time.sleep(1)
    try:
        while True:
            time.sleep(2)
            start = time.time()

            frames = q.pop()
            temp_file_name = os.path.join("tmp", "tmp-" + str(time.time()) + ".wav")
            predicted = 0
            try:
                wf = wave.open(temp_file_name, 'wb')
                wf.setnchannels(recorder.CHANNELS)
                wf.setsampwidth(recorder.p.get_sample_size(recorder.FORMAT))
                wf.setframerate(recorder.RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                binary = tf.io.read_file(temp_file_name)
                decoded, _ = tf.audio.decode_wav(binary, desired_channels=1, desired_samples=44100)
                predicted = model.predict(np.array([decoded]))[0][0]
                print(f"Predicted {predicted}. Took: {time.time() - start} seconds")

            finally:
                if predicted < 0.5:
                    os.remove(temp_file_name)

    except:
        print("Stopping")
        recorder.stop()
        raise
