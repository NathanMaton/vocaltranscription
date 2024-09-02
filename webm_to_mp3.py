import subprocess
import sys
import os
from pathlib import Path

def convert_webm_to_mp3(input_file, data_folder):
    converted_audio_folder = os.path.join(data_folder, 'converted_audio')
    os.makedirs(converted_audio_folder, exist_ok=True)

    output_file = os.path.join(converted_audio_folder, input_file.stem + '.mp3')
    print(f"Converting {input_file} to {output_file}")
    try:
        subprocess.run(['ffmpeg', '-i', str(input_file), '-vn', '-acodec', 'libmp3lame', '-b:a', '192k', output_file], check=True)
        print(f"Conversion complete. Output saved as '{output_file}'")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and available in your system PATH.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python webm_to_mp3.py input.webm")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    data_folder = 'data'

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    if not input_file.suffix.lower() == '.webm':
        print("Error: Input file must be a .webm file.")
        sys.exit(1)

    convert_webm_to_mp3(input_file, data_folder)