# Sound classification

This is simple weekend project to recognize and record sounds in real time with tensorflow an keras. It has 3 python entrypoints:

- `real_time_classifier.py`: It constantly records 2 second audio clips and runs a RNN against it and tries to classify whether the trained sound is there or not. If the confidence is greather or equal than 0.5 it saves a wav file in `training-data/recognized/` folder.
- `train.py`: It picks all recordings from `training-data/positive` and `training-data/negative`, trains a basic RNN and saves the output in `model.tflite`
- `web.py`: It starts a flask app with some useful tools to collect training data and analyze the real time recordings. It has the following paths:
  - `/recorder`: The server in background constantly records sounds withing a 30 second window and this page contains a submit button to save a wav file with the recorder buffer into `training-data/raw/`. Later, those files can be trimmed and used as training data.
  - `/cropper`: It shows a page with all the raw recordings and allows to extract 2 seconds clips and mark them as positive sounds
  - `/analyze`: it shows a page with all the recognized sounds by the `real_time_classifier` and allows to mark them as `positive` or `negative` which can be later used for training
  - `/report`: It shows a simple json with the count of positive sounds by date

## Docker usage

Build `docker build -t classifier .`

Run: `docker run --rm -d --device /dev/snd -v "$(pwd)":/app classifier python real_time_classifier.py`


## TODO

- https://towardsdatascience.com/how-to-reduce-training-time-for-a-deep-learning-model-using-tf-data-43e1989d2961
- https://stackoverflow.com/questions/54431168/how-to-cache-layer-activations-in-keras
- https://github.com/kongkip/spela
- https://medium.com/swlh/how-to-run-gpu-accelerated-signal-processing-in-tensorflow-13e1633f4bfb



- https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06