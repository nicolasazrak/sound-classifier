import os
from utils import get_raw_recordings


def generate_background():
    for recording in get_raw_recordings():
        os.system(f'ffmpeg -i "training-data/raw/{recording}" -f segment -segment_time 2 -c copy "training-data/background/{recording}%03d.wav"')


generate_background()
