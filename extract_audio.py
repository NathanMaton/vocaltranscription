import yt_dlp
import time
import re
import os
import subprocess

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\.]', '', filename.replace(' ', '_'))

def extract_audio(data_folder):
    video_url = input("Enter the YouTube URL: ")
    cut_audio = input("Do you want to cut the audio? (y/n): ").lower() == 'y'
    if cut_audio:
        cut_duration = input("Enter duration in seconds (15 or 90): ")
        if cut_duration not in ['15', '90']:
            print("Invalid duration. Defaulting to full audio.")
            cut_duration = None
        else:
            cut_duration = int(cut_duration)
    else:
        cut_duration = None
    
    extracted_audio_folder = os.path.join(data_folder, 'extracted_audio')
    os.makedirs(extracted_audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': {'default': os.path.join(extracted_audio_folder, '%(title)s.%(ext)s')},
    }
    
    print(f"Extracting audio from: {video_url}")
    start_time = time.time()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            title = info['title']
            sanitized_title = sanitize_filename(title)
            output_path = os.path.join(extracted_audio_folder, f'{sanitized_title}.mp3')
            ydl.params['outtmpl']['default'] = output_path
            ydl.download([video_url])
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return

    # Check for double extension and correct it
    if os.path.exists(output_path + '.mp3'):
        os.rename(output_path + '.mp3', output_path)

    if not os.path.exists(output_path):
        print(f"Error: The expected output file {output_path} was not created.")
        return

    if cut_duration:
        temp_output_path = os.path.join(extracted_audio_folder, f'{sanitized_title}_{cut_duration}sec.mp3')
        ffmpeg_command = [
            'ffmpeg', '-i', output_path, '-t', str(cut_duration), '-acodec', 'copy', temp_output_path
        ]
        try:
            subprocess.run(ffmpeg_command, check=True, stderr=subprocess.PIPE, text=True)
            os.replace(temp_output_path, output_path)
            print(f"Audio cut to {cut_duration} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"Error cutting audio: {e.stderr}")
            return
    
    print(f"Audio extraction complete. Output saved as: {output_path}")
    print(f"Extraction took {time.time() - start_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    data_folder = 'data'
    extract_audio(data_folder)