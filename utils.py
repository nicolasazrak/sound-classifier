import os
import librosa
import numpy as np


def get_raw_recordings():
    return get_recordings('raw')


def get_recognized_recordings():
    return get_recordings('recognized')


def get_positive():
    positive = []
    positive.extend(get_recordings('positive-clean'))
    positive.extend(get_recordings('positive-noisy'))
    return positive


def get_recordings(from_type):
    recordings = []
    for file in os.listdir(os.path.join("training-data", from_type)):
        if file.endswith(".wav"):
            recordings.append(file)
    recordings.sort()
    return recordings


def generate_thumb(file_path):
    if not os.path.isfile(file_path + ".png"):
        print("Generating {file_path}.png")
        os.system(
            f'ffmpeg -i "{file_path}"  -filter_complex "[0:a]aformat=channel_layouts=mono,  compand=gain=10,  showwavespic=s=600x120:colors=#9cf42f[fg];  color=s=600x120:color=#44582c,  drawgrid=width=iw/10:height=ih/5:color=#9cf42f@0.1[bg];  [bg][fg]overlay=format=auto,drawbox=x=(iw-w)/2:y=(ih-h)/2:w=iw:h=1:color=#9cf42f" -frames:v 1 "{file_path}.png"')


def generate_thumbs():
    for recording in get_raw_recordings():
        file_path = os.path.join("training-data", "raw", recording)
        generate_thumb(file_path)
    for recording in get_recognized_recordings():
        file_path = os.path.join("training-data", "recognized", recording)
        generate_thumb(file_path)


def generate_background():
    for recording in get_raw_recordings():
        os.system(f'ffmpeg -i "training-data/raw/{recording}" -f segment -segment_time 2 -c copy "training-data/background/{recording}%03d.wav"')


def load_audio(file_or_path):
    padded = np.zeros((44100,))
    y, sr = librosa.load(file_or_path, sr=22050, duration=2)
    padded[:y.shape[0]] = y[:]
    spect = librosa.feature.melspectrogram(y=padded, sr=sr)
    swapped = np.swapaxes(spect, 0, 1)
    return swapped
