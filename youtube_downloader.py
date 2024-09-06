import yt_dlp
import os

def download_youtube_video(url, filename):
    output_path = 'data/videos'
    os.makedirs(output_path, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_path, f'{filename}.%(ext)s'),
        'merge_output_format': 'mp4',
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
        print(f"Download complete! File saved to: {filename}")
        return filename
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Get the YouTube URL from user input
    video_url = input("Enter the YouTube video URL: ")
    
    # Get the desired filename from user input
    filename = input("Enter the desired filename (without extension): ")
    
    # Download the video
    downloaded_file = download_youtube_video(video_url, filename)
    
    if downloaded_file:
        print(f"Video downloaded successfully: {downloaded_file}")
    else:
        print("Failed to download the video.")