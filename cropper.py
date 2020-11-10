import wave
import threading
import sys
import time
import os
import os.path
import numpy as np
import sys
import random
from utils import get_raw_recordings, generate_thumbs
from flask import Flask, render_template, request


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
    outname = recording.split(".wav")[0] + "-" + str(random.randint(0, 1000000)) + ".wav"
    os.system(f'ffmpeg -ss {position} -i "training-data/raw/{recording}" -t 2 "training-data/positive-{crop_type}/{outname}"')

    return "OK"


# app.run(debug=True, host='0.0.0.0')
app.run(debug=True)
# ffmpeg -i somefile.mp3 -f segment -segment_time 3 -c copy out%03d.mp3
# https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple
