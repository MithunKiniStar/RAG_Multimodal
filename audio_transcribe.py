# -*- coding: utf-8 -*-
"""
AudioTranscribe Script

This script performs the following tasks:
1. Converts a video file to MP3 format using ffmpeg.
2. Transcribes the MP3 file using OpenAI's Whisper model.
"""

import os
import subprocess
import whisper  # Ensure this is OpenAI's Whisper

# --------------------------
# Helper Functions
# --------------------------

def convert_video_to_mp3(input_file, output_file):
    """Converts a video file to an MP3 audio file using ffmpeg.

    Args:
        input_file (str): Path to the input video file.
        output_file (str): Path to the output MP3 file.
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vn",  # No video
        "-acodec", "libmp3lame",
        "-ab", "192k",  # Bitrate
        "-ar", "44100",  # Sample rate
        "-y",  # Overwrite output
        output_file
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Conversion successful: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed: {e}")
        raise

def transcribe_audio(audio_file, model_name="medium.en"):
    """Transcribes an audio file using Whisper Python API.

    Args:
        audio_file (str): Path to the MP3 audio file.
        model_name (str): Whisper model to use for transcription (default: "medium.en").
    """
    try:
        # Load the Whisper model
        model = whisper.load_model(model_name)
        # Transcribe the audio file
        result = model.transcribe(audio_file)
        # Save transcription to a text file
        output_file = f"{audio_file}.txt"
        with open(output_file, "w") as f:
            f.write(result["text"])
        print(f"Transcription completed: {output_file}")
    except Exception as e:
        print(f"Transcription failed: {e}")
        raise

# --------------------------
# Main Execution Workflow
# --------------------------

def main():
    """Main workflow for the script."""
    # Define file paths
    input_video = "./source/sales_force_demo.mp4"
    output_audio = "./source/sales_force_demo.mp3"

    # Convert video to audio
    print("Starting video-to-audio conversion...")
    convert_video_to_mp3(input_video, output_audio)

    # Transcribe the audio
    print("Starting audio transcription...")
    transcribe_audio(output_audio)

if __name__ == "__main__":
    main()