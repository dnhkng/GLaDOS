#!/bin/bash

# Twitch Streaming Configuration
TWITCH_STREAM_KEY=""  # Replace with your Twitch stream key
# Replace with your Twitch ingest server (cf: https://help.twitch.tv/s/twitch-ingest-recommendation?language=en_US)
TWITCH_INGEST_SERVER="rtmp://ingest.global-contribute.live-video.net/app/" 
# Command for launching glados UI
COMMAND="python -m src.glados.cli tui --mode twitch"
# Resolution and FPS for video streaming
WIDTH=1280
HEIGHT=720
FPS=30


# Validate Twitch stream key
if [ -z "$TWITCH_STREAM_KEY" ]; then
    echo "Error: TWITCH_STREAM_KEY is not set. Please provide a valid stream key."
    exit 1
fi

# Ensure the script is run from the root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$(pwd)" != "$ROOT_DIR" ]; then
    echo "This script must be run from the root directory of the project."
    echo "Changing to the root directory: $ROOT_DIR"
    cd "$ROOT_DIR" || exit 1
fi


# Get default audio device for PulseAudio
AUDIO_INPUT=$(pactl info | grep 'Default Source' | cut -d':' -f2 | xargs)

if [ -z "$AUDIO_INPUT" ]; then
    echo "Warning: Could not find default audio source. Using default audio input."
    AUDIO_INPUT="default"
else
    echo "Using audio source: $AUDIO_INPUT"
fi

echo "Starting FFmpeg for streaming..."

# Start the application directly
echo "Starting application with command: $COMMAND"
$COMMAND &

# Start streaming with FFmpeg (capture the virtual display and system audio)
echo "Starting glados with command: $COMMAND"
ffmpeg \
    -f x11grab -draw_mouse 0 -framerate ${FPS} -video_size ${WIDTH}x${HEIGHT} -i ${DISPLAY} \
    -f pulse -ac 2 -i "${AUDIO_INPUT}" \
    -c:v libx264 -preset veryfast -b:v 3000k -maxrate 3500k -bufsize 6000k \
    -pix_fmt yuv420p \
    -g 60 -keyint_min 30 \
    -c:a aac -b:a 128k -ar 44100 \
    -f flv "${TWITCH_INGEST_SERVER}/${TWITCH_STREAM_KEY}"