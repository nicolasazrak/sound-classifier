import wave
import threading
import sys
import time
import os
import os.path
import numpy as np
import sys
import random
from utils import get_recognized_recordings, generate_thumbs
from flask import Flask, render_template, request


generate_thumbs()
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('analyzer.html', recordings=get_recognized_recordings())


@app.route('/confirm', methods=['POST'])
def crop():
    recording = request.json['recording']
    crop_type = request.json['type']
    outname = recording.split(".wav")[0] + "-" + str(random.randint(0, 1000000)) + ".wav"
    os.system(f'mv "training-data/recognized/{recording}" "training-data/{crop_type}/{recording}"')
    os.system(f'rm "training-data/recognized/{recording}.png"')

    return "OK"


app.run(debug=True)
