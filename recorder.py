import sys
import wave
import time
import struct
import pyaudio
import threading
import os.path
import numpy as np


class BufferedRecorder:

    def __init__(self):
        self.buffer = []
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            start=False,
            stream_callback=self.on_audio
        )

    def run(self):
        self.strea.start_stream()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

    def save_last_30_seconds(self, file_name):
        output_name = os.path.join("training-data", "raw", "output-" + str(time.now()) + ".wav")
        save_wav(output_name, self.buffer)

    def on_audio(self, in_data, frame_count, time_info, status):
        new_samples = np.frombuffer(in_data, dtype=np.float32)
        self.buffer.extend(new_samples)
        self.buffer = self.samples_buffer[-16000 * 30:]
        return None, pyaudio.paContinue


def save_wav(file_name, samples):
    with wave.open(file_name, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        s = np.array(samples, dtype=np.float32)
        s = s * (2 ** 15)
        s = s.astype(int)
        data = struct.pack('h' * len(s), *s)
        f.writeframes(data)
        # f.writeframes(s.byteswap().tobytes())


def record(callback):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        stream_callback=callback
    )

    try:
        stream.start_stream()
        while True:
            time.sleep(1)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
