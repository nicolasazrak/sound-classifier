# Sound classification

This is a real-time audio classification project using TensorFlow and YamNet. It detects specific sounds (like barks) using deep learning and provides tools for collecting and managing training data.

## Features

- **Real-time detection**: Processes 2-second audio clips continuously
- **Data collection**: 30-second buffer with manual save functionality
- **Training pipeline**: Simple RNN classifier on YamNet embeddings
- **Web interface**: Flask-based tools for data management and analysis

## Entry Points

- `real_time_classifier.py`: Continuously records 2-second audio clips and classifies them. Saves clips with confidence ≥ 0.5 to `training-data/recognized/`
- `train.py`: Trains a classifier using recordings from `training-data/positive` and `training-data/negative`, saves model to `model.tflite`
- `web.py`: Flask webapp for data collection and analysis with these routes:
  - `/recorder`: 30-second audio buffer with save button to `training-data/raw/`
  - `/cropper`: Extract 2-second clips from raw recordings and mark as positive
  - `/analyze`: Review recognized sounds and mark as positive/negative for training
  - `/report`: JSON report of positive sounds by date

## Quick Start

### Using UV (Recommended)

```bash
# Install dependencies
uv sync

# Run applications
uv run python real_time_classifier.py  # Real-time classification
uv run python web.py                    # Web interface
uv run python train.py                  # Train model (requires full TensorFlow)
```

### Raspberry Pi Setup

For Raspberry Pi and other ARM devices, use the optimized TensorFlow Lite runtime:

```bash
# Install dependencies (optimized for ARM)
uv sync

# The project uses tflite-runtime for inference on Raspberry Pi
# Training requires full TensorFlow and should be done on a more powerful system
```


## Docker Usage

Build and run:
```bash
docker build -t sound-classifier .

# Run classifier with audio device access
docker run --rm -d --device /dev/snd -v "$(pwd)":/app sound-classifier python real_time_classifier.py

# Run web app
docker run --rm -d --device /dev/snd -v "$(pwd)":/app -p 5000:5000 sound-classifier python web.py
```

## Architecture

- **Model**: YamNet (pretrained on AudioSet) for feature extraction + RNN classifier
- **Audio Processing**: PyAudio for real-time recording, 44.1kHz → 16kHz resampling
- **Inference**: TensorFlow Lite for edge deployment
- **Web**: Flask with HTML templates for data management

## Project Structure

```
sound-classifier/
├── training-data/          # Audio samples
│   ├── positive/          # Positive training samples
│   ├── negative/          # Negative training samples
│   ├── raw/               # Raw recordings for processing
│   └── recognized/        # Auto-detected sounds
├── static/                # Web assets
├── templates/             # HTML templates
├── pyproject.toml         # UV project configuration
├── real_time_classifier.py # Main detection script
├── train.py              # Training script
├── web.py                # Web application
└── recorder.py           # Audio recording utilities
```


### Code Formatting
```bash
uv run black .
uv run ruff check .
```

## System Requirements

- Python 3.11
- Audio input device (microphone)
- For Docker: Linux with audio device support
- Recommended: 2GB+ RAM for TensorFlow
- **Raspberry Pi**: Use `tflite-runtime` package (included by default)
- **Training**: Full TensorFlow required (install with `uv sync --extras train`)


## TODO

- https://towardsdatascience.com/how-to-reduce-training-time-for-a-deep-learning-model-using-tf-data-43e1989d2961
- https://stackoverflow.com/questions/54431168/how-to-cache-layer-activations-in-keras
- https://github.com/kongkip/spela
- https://medium.com/swlh/how-to-run-gpu-accelerated-signal-processing-in-tensorflow-13e1633f4bfb
- https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
