import sys
import wave
import time
import os
import os.path
import wave
import random
import itertools
import threading
import numpy as np
from recorder import Recorder
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)


@app.route('/analyze')
def view_analyze_page():
    generate_thumbs()
    return render_template('analyzer.html', recordings=list_recognized_recordings())


@app.route('/report')
def view_report_page():
    positive = list_positive_recordings()
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

    return jsonify(counts)


@app.route('/analyze/confirm', methods=['POST'])
def confirm_recording():
    recording = request.json['recording']
    crop_type = request.json['type']
    outname = recording.split(".wav")[0] + "-" + str(random.randint(0, 1000000)) + ".wav"
    os.system(f'mv "training-data/recognized/{recording}" "training-data/{crop_type}/{recording}"')
    os.system(f'rm "training-data/recognized/{recording}.png"')

    return "OK"


@app.route('/crop', methods=['GET'])
def view_cropper():
    return render_template('cropper.html', recordings=list_raw_recordings())


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
def view_recorder_page():
    return render_template('recorder.html')


@app.route('/recordings', methods=["POST"])
def save_recording():
    return recorder.save_last_30_seconds()


try:
    recorder = BufferedRecorder()
    recorder.run()

    app.run(debug=True, host='0.0.0.0')
finally:
    recorder.stop()


# Utils


def list_raw_recordings():
    return list_recordings('raw')


def list_recognized_recordings():
    return list_recordings('recognized')


def list_positive_recordings():
    return list_recordings('positive')


def list_recordings(from_type):
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
    for recording in list_raw_recordings():
        file_path = os.path.join("training-data", "raw", recording)
        generate_thumb(file_path)
    for recording in list_recognized_recordings():
        file_path = os.path.join("training-data", "recognized", recording)
        generate_thumb(file_path)


def split_raw_as_negative():
    for recording in list_raw_recordings():
        os.system(f'ffmpeg -i "training-data/raw/{recording}" -f segment -segment_time 2 -c copy "training-data/negative/{recording}%03d.wav"')
