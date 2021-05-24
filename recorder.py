import sys
import wave
import time
import struct
import pyaudio
import threading
import os.path
import scipy.signal as signal
import numpy as np


class Recording:

    def __init__(self, duration, bytes_chunks, original_sample_rate):
        self.duration = duration
        self.bytes_chunks = bytes_chunks
        self.original_sample_rate = original_sample_rate

    def samples_at(self, rate):
        new_samples = np.frombuffer(self.bytes_chunks, dtype=np.float32)
        return signal.resample(new_samples, rate * self.duration)

    def save_to(self, file_name):
        s = np.frombuffer(self.bytes_chunks, dtype=np.float32)
        s = s * (2 ** 15 - 1)
        s = s.astype(np.int16)
        with wave.open(file_name, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(self.original_sample_rate)
            s.byteswap()
            f.writeframes(s.tobytes())


class BufferedRecorder:

    def __init__(self, buffer_seconds):
        self.buffer = bytearray()
        self.rate = 44100
        self.buffer_seconds = buffer_seconds
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            start=False,
            frames_per_buffer=4096,
            stream_callback=self.on_audio
        )

    def run(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

    def get_last_30_seconds_recording(self):
        return Recording(self.buffer_seconds, self.buffer, self.rate)

    def on_audio(self, in_data, frame_count, time_info, status):
        self.buffer.extend(in_data)
        self.buffer = self.buffer[-self.rate * self.buffer_seconds * 4:]
        return None, pyaudio.paContinue


class ChunkedRecorder:

    def __init__(self, recording_duration, callback):
        self.buffer = bytearray()
        self.rate = 44100
        self.recording_duration = recording_duration
        self.p = pyaudio.PyAudio()
        self.callback = callback
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.rate,
            input=True,
            stream_callback=self._on_audio
        )

    def _on_audio(self, in_data, frame_count, time_info, status):
        self.buffer.extend(in_data)
        required_samples = self.rate * self.recording_duration * 4  # 4 bytes each sample
        while len(self.buffer) > required_samples:
            samples_for_recording, remaining = self.buffer[:required_samples], self.buffer[required_samples:]
            self.buffer = remaining
            self.callback(Recording(self.recording_duration, samples_for_recording, self.rate))
        return None, pyaudio.paContinue

    def run(self):
        try:
            self.stream.start_stream()
            while True:
                time.sleep(1)
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


if __name__ == "__main__":
    def cb(recording):
        print(len(recording.samples_at(16000)))

    # c = ChunkedRecorder(2, cb)
    # c.run()

    c = BufferedRecorder(5)
    c.run()
    time.sleep(3)
    r = c.get_last_30_seconds_recording()
    r.save_to('hola.wav')
