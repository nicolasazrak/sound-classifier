import sys
import wave
import time
import os
import os.path
import random
import itertools
import threading
from recorder import Recorder
from flask import Flask, render_template, request
from utils import get_recognized_recordings, generate_thumbs, get_positive, get_raw_recordings


app = Flask(__name__)


@app.route('/analyze')
def analyze():
    generate_thumbs()
    return render_template('analyzer.html', recordings=get_recognized_recordings())


@app.route('/report')
def report():
    positive = get_positive()
    positive = map(lambda r: r.split("-")[1].split(".wav")[0], positive)
    positive = map(lambda r: time.gmtime(float(r)), positive)

    grouped = {}
    counts = {}
    for elem in positive:
        d = f"{elem.tm_year}-{elem.tm_mon}-{elem.tm_mday}"
        if not d in grouped:
            grouped[d] = []
            counts[d] = 0
        grouped[d].append(elem)
        counts[d] += 1

    return "OK"


@app.route('/analyze/confirm', methods=['POST'])
def confirm_recording():
    recording = request.json['recording']
    crop_type = request.json['type']
    outname = recording.split(".wav")[0] + "-" + str(random.randint(0, 1000000)) + ".wav"
    os.system(f'mv "training-data/recognized/{recording}" "training-data/{crop_type}/{recording}"')
    os.system(f'rm "training-data/recognized/{recording}.png"')

    return "OK"


@app.route('/crop', methods=['GET'])
def view_crop():
    return render_template('cropper.html', recordings=get_raw_recordings())


@app.route('/crop', methods=['POST'])
def crop_audio():
    # ffmpeg -i somefile.mp3 -f segment -segment_time 3 -c copy out%03d.mp3
    # https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple
    position = request.json['position']
    recording = request.json['recording']
    outname = recording.split(".wav")[0] + "-" + str(random.randint(0, 1000000)) + ".wav"
    os.system(f'ffmpeg -ss {position} -i "training-data/raw/{recording}" -t 2 "training-data/positive/{outname}"')

    return "OK"


@app.route('/recorder')
def view_recorder():
    return render_template('recorder.html')


@app.route('/recordings', methods=["POST"])
def save_recording():
    return recorder.save_buffer()


try:
    recorder = Recorder(record_seconds=30)
    threading.Thread(target=recorder.start).start()

    app.run(debug=True, host='0.0.0.0')
finally:
    recorder.stop()
