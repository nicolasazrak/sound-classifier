#!/usr/bin/env python3
"""
Audio device testing utility for sound classifier.
Helps identify the best audio device for recording and tests audio capture.
"""

import pyaudio
import time
import numpy as np
from recorder import BufferedRecorder


def list_audio_devices():
    """List all available audio devices and their capabilities."""
    p = pyaudio.PyAudio()
    print("=== Available Audio Devices ===\n")

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
        print(f"  Max input channels: {info['maxInputChannels']}")
        print(f"  Default sample rate: {info['defaultSampleRate']}")

        if info["maxInputChannels"] > 0:
            # Test sample rate compatibility
            for rate in [44100, 16000, 48000]:
                try:
                    stream = p.open(
                        format=pyaudio.paFloat32,
                        channels=1,
                        rate=rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024,
                    )
                    print(f"  ✓ Supports {rate} Hz")
                    stream.close()
                except Exception as e:
                    print(f"  ✗ {rate} Hz: {str(e)[:50]}")
        print()

    p.terminate()


def test_audio_capture(device_index=None, duration=2):
    """Test actual audio capture from a specific device."""
    print(f"\n=== Testing Audio Capture (Device {device_index}) ===")
    print(f"Recording for {duration} seconds...")
    print("(Make some noise or speak into the microphone)")

    try:
        c = BufferedRecorder(buffer_seconds=duration, input_device=device_index)
        c.run()
        time.sleep(duration)

        r = c.get_recoding_from_last(seconds=duration)
        c.stop()

        # Test at both sample rates
        for rate in [44100, 16000]:
            samples = r.samples_at(rate)
            print(f"\n{rate} Hz:")
            print(f"  Shape: {samples.shape}")
            print(f"  Range: [{samples.min():.6f}, {samples.max():.6f}]")
            print(f"  Mean: {samples.mean():.6f}")
            print(f"  Std: {samples.std():.6f}")
            print(f"  Non-zero: {np.count_nonzero(samples)}/{len(samples)}")

            # Determine if audio was captured
            if samples.std() > 0.001:
                print(f"  ✓ Audio captured successfully!")
            else:
                print(f"  ✗ No audio detected (silent or muted)")

            # Save test file
            filename = f"test_audio_{rate}Hz.wav"
            r.save_to_wav(filename, rate)
            print(f"  Saved to: {filename}")

        return True

    except Exception as e:
        print(f"\n✗ Error during audio capture: {e}")
        return False


def find_best_device():
    """Find the best audio device for recording."""
    p = pyaudio.PyAudio()
    best_device = None
    best_score = -1

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            score = 0

            # Prefer devices that support 44100 Hz
            try:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=1024,
                )
                stream.close()
                score += 10
            except:
                pass

            # Prefer devices that support 16000 Hz
            try:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=16000,
                    input=True,
                    input_device_index=i,
                    frames_per_buffer=1024,
                )
                stream.close()
                score += 5
            except:
                pass

            # Prefer lower latency
            if info["defaultLowInputLatency"] < 0.02:
                score += 3

            # Prefer devices with "default" or "analog" in name
            name_lower = info["name"].lower()
            if "default" in name_lower or "analog" in name_lower:
                score += 2

            print(f"Device {i} ({info['name']}): Score {score}")

            if score > best_score:
                best_score = score
                best_device = i

    p.terminate()

    return best_device


def main():
    """Main testing function."""
    print("Sound Classifier - Audio Device Test Utility")
    print("=" * 50)

    # List all devices
    list_audio_devices()

    # Find best device
    print("\n=== Finding Best Device ===")
    best_device = find_best_device()

    if best_device is not None:
        print(f"\n✓ Recommended device: {best_device}")

        # Test the recommended device
        answer = input(f"\nDo you want to test device {best_device}? (y/n): ")
        if answer.lower().startswith("y"):
            test_audio_capture(device_index=best_device)

        # Optionally test specific device
        answer = input("\nDo you want to test a specific device? (number or n): ")
        if answer.lower() != "n":
            try:
                device_num = int(answer)
                test_audio_capture(device_index=device_num)
            except ValueError:
                print("Invalid device number")
    else:
        print("\n✗ No suitable audio device found")
        print("\nTroubleshooting:")
        print("1. Check if microphone is connected")
        print("2. Check audio permissions")
        print("3. Check if audio drivers are installed")
        print("4. Try running with sudo (Linux)")


if __name__ == "__main__":
    main()
