import pyaudio
import wave
import threading
import sys
import time
import os.path


class Recorder:

    def __init__(self, record_seconds):
        self.p = pyaudio.PyAudio()
        self.frames = []
        self.should_stop = False
        self.buffer_lock = threading.Lock()
        self.stop_lock = threading.Lock()
        self.record_seconds = record_seconds

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

    def get_last_samples(self):
        frames_copy = []
        self.buffer_lock.acquire()
        try:
            frames_copy = self.frames.copy()
        finally:
            self.buffer_lock.release()
        return frames_copy

    def save_buffer(self):
        output_name = os.path.join("training-data", "raw", "output-" + str(time.time()) + ".wav")
        wf = wave.open(output_name, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        frames_copy = self.get_last_samples()
        wf.writeframes(b''.join(frames_copy))
        wf.close()

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

        buffer_frame_size = int(self.RATE / self.CHUNK * self.record_seconds)

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
