#!/usr/bin/env python3
"""
Configuration for sound classifier audio settings.
"""

INPUT_DEVICE = 7
SAMPLE_RATE = 44100  # Must match your audio device's native rate

# Web app configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}

# Real-time classifier configuration
CLASSIFIER_CONFIG = {
    "confidence_threshold": 0.5,
    "model_path": "model.tflite",
    "yamnet_path": "yamnet.tflite",
}
