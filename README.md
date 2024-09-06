# Music Video Production Tools

This repository contains several Python scripts to assist in music video production:

1. `video_chopper.py`
2. `music_video_storyboard.py`
3. `vocal_parts_to_sheet_music.py`
4. `vocal_to_sheet_music.py`
5. `extract_audio.py`
6. `webm_to_mp3.py`
7. `youtube_downloader.py` -OLD, USE EXTRACT AUDIO INSTEAD

## video_chopper.py

This script chops a video into multiple segments of equal duration. It's useful for creating short clips from a longer video, which can be used in music video production or social media content.

Key features:
- Splits a video into a specified number of segments with a given duration
- Uses FFmpeg for video processing
- Outputs segments in MP4 format
- Clears existing files in the output folder before processing

## music_video_storyboard.py

This script analyzes an audio file and generates a storyboard spreadsheet for music video production. It detects major transitions in the song and provides a framework for planning the video.

Key features:
- Analyzes audio using librosa to detect major transitions
- Generates timestamps for significant moments in the song
- Creates an Excel spreadsheet with columns for Timestamp, Lyrics, Image Ideas, and Vibes
- Provides a simple "vibe" generation based on the song's structure
- Includes overall tempo information

## vocal_parts_to_sheet_music.py

This script converts vocal parts from an M4A audio file to sheet music.

Key features:
- Converts M4A to WAV format
- Splits vocal parts based on energy peaks
- Transcribes vocal parts to sheet music using pitch detection
- Outputs sheet music in MusicXML format

## vocal_to_sheet_music.py

This script processes an audio file to create sheet music from vocal parts.

Key features:
- Extracts vocals from the audio
- Performs pitch detection and quantization
- Creates sheet music with key detection and time signature
- Outputs sheet music in MusicXML format and MIDI file

## extract_audio.py

This script extracts audio from a YouTube video.

Key features:
- Downloads the best quality audio from a YouTube video
- Converts the audio to MP3 format
- Sanitizes the output filename

## webm_to_mp3.py

This script converts WebM audio files to MP3 format.

Key features:
- Uses FFmpeg for audio conversion
- Outputs high-quality MP3 files

## youtube_downloader.py

This script downloads YouTube videos in the highest available quality.

Key features:
- Downloads the best quality video and audio
- Merges video and audio into MP4 format
- Allows custom output filenames

## Dependencies

To run these scripts, you'll need to install various Python packages and external tools. Please refer to each script for specific dependencies.

## File Structure

- `data/videos/`: Input/output folder for video files
- `data/video_segments/`: Output folder for chopped video segments
- `data/audio/`: Input folder for audio files
- `data/storyboards/`: Output folder for storyboard spreadsheets
- `data/extracted_audio/`: Output folder for extracted audio from YouTube
- `data/converted_audio/`: Output folder for converted audio files

## Note

These scripts provide a comprehensive framework for various music video production tasks. You may want to customize them further based on your specific needs.