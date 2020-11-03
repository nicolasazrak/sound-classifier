import pyaudio
import wave
import threading
import sys
import time
import os
import os.path
import numpy as np
import sys
import random
from flask import Flask, render_template, request


def get_raw_recordings():
    recordings = []
    for file in os.listdir("training-data/raw/"):
        if file.endswith(".wav"):
            recordings.append(file)
    recordings.sort()
    return recordings


def generate_thumbs():
    for recording in get_raw_recordings():
        if not os.path.isfile(os.path.join("training-data", "raw", recording + ".png")):
            print("Generating training-data/raw/{recording}.png")
            os.system(
                f'ffmpeg -i "training-data/raw/{recording}"  -filter_complex "[0:a]aformat=channel_layouts=mono,  compand=gain=10,  showwavespic=s=600x120:colors=#9cf42f[fg];  color=s=600x120:color=#44582c,  drawgrid=width=iw/10:height=ih/5:color=#9cf42f@0.1[bg];  [bg][fg]overlay=format=auto,drawbox=x=(iw-w)/2:y=(ih-h)/2:w=iw:h=1:color=#9cf42f" -frames:v 1 "recordings/raw/{recording}.png"')


# def generate_background():
#    for recording in get_raw_recordings():
#        os.system(
#            f'ffmpeg -i "recordings/raw/{recording}" -f segment -segment_time 2 -c copy "training-data/background/{recording}%03d.wav"')
# generate_background()

generate_thumbs()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('cropper.html', recordings=get_raw_recordings())


@app.route('/crop', methods=['POST'])
def crop():
    position = request.json['position']
    recording = request.json['recording']
    crop_type = request.json['type']
    outname = recording.split(
        ".wav")[0] + "-" + str(random.randint(0, 1000000)) + ".wav"
    os.system(
        f'ffmpeg -ss {position} -i "training-data/raw/{recording}" -t 2 "training-data/positive-{crop_type}/{outname}"')
    return "OK"


# app.run(debug=True, host='0.0.0.0')
app.run(debug=True)
# ffmpeg -i somefile.mp3 -f segment -segment_time 3 -c copy out%03d.mp3
# https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple
