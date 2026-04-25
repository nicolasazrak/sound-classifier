# docker run --rm -i -t --device /dev/snd -v "$(pwd)":/app sound-classifier bash
# arecord -d 10 -D plughw:2,0 -f S16_LE --rate=16000  --channels=1 test.wav

FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    portaudio19-dev \
    gcc \
    alsa-utils \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add UV to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Create app directory
RUN mkdir /app
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY . ./

# Install dependencies using UV
RUN uv sync --frozen

# For production on ARM64 (Raspberry Pi), install TensorFlow Lite runtime
# This is optional as the full TensorFlow package can also use tflite
# Uncomment if you need the lightweight tflite_runtime package
# RUN uv pip install --system --index-url https://www.piwheels.org/simple tflite-runtime

# Default command
CMD ["uv", "run", "python", "real_time_classifier.py"]
