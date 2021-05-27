# docker run --rm -i -t --device /dev/snd -v "$(pwd)":/app sound-classifier bash
# arecord -d 10 -D plughw:2,0 -f S16_LE --rate=16000  --channels=1 test.wav
FROM python:3.7.10-slim-buster

# RUN apt-get update && apt-get install -y curl gnupg portaudio19-dev python3-pyaudio python3-tflite-runtime gcc
RUN apt-get update && apt-get install -y curl gnupg portaudio19-dev gcc alsa-utils nano
RUN pip install pyaudio numpy scipy flask
RUN mkdir /app
WORKDIR /app
ADD . .

# Taken from https://github.com/PINTO0309/TensorflowLite-bin/blob/master/2.3.1/download_tflite_runtime-2.3.1-py3-none-linux_aarch64.whl.sh
RUN pip install tflite_runtime-2.3.1-py3-none-linux_aarch64.whl 