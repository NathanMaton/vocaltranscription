import yt_dlp
import time
import re
import os

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_\.]', '', filename.replace(' ', '_'))

def extract_audio(video_url, data_folder):
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
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        title = info['title']
        sanitized_title = sanitize_filename(title)
        output_path = os.path.join(extracted_audio_folder, f'{sanitized_title}.%(ext)s')
        ydl.params['outtmpl']['default'] = output_path
        ydl.download([video_url])
    
    print(f"Audio extraction complete. Output saved as: {os.path.join(extracted_audio_folder, sanitized_title)}.mp3")
    print(f"Extraction took {time.time() - start_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    video_url = 'https://www.youtube.com/watch?v=Ta_NFLtu920'
    data_folder = 'data'
    extract_audio(video_url, data_folder)