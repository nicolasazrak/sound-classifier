import threading
from flask import Flask, render_template
from recorder import Recorder

app = Flask(__name__)

recorder = Recorder()
threading.Thread(target=recorder.start).start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/save', methods=["POST"])
def save():
    return recorder.save_buffer()


try:
    app.run(debug=True)
finally:
    recorder.stop()
