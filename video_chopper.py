import os
import subprocess

def chop_video(video_path, output_folder, segment_duration=5, num_segments=3):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Clear existing files in the output folder
    for file in os.listdir(output_folder):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    # Get the video filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i in range(num_segments):
        start_time = i * segment_duration
        
        # Generate output filename with segment length
        output_filename = f"{base_name}_segment_{i+1:03d}_{segment_duration}sec.mp4"
        output_path = os.path.join(output_folder, output_filename)

        # FFmpeg command
        command = [
            'ffmpeg',
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(segment_duration),
            '-c:v', 'libx264',  # Use H.264 codec for video
            '-c:a', 'aac',      # Use AAC codec for audio
            '-strict', 'experimental',
            output_path
        ]

        try:
            # Execute FFmpeg command
            subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Segment {i+1} saved: {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"Error writing segment {i+1}: {e.stderr}")

if __name__ == "__main__":
    # Input video filename
    video_filename = input("Enter the filename of the video in data/videos folder (including extension): ")
    
    # Construct full path to input video
    input_path = os.path.join("data", "videos", video_filename)

    # Output folder for segments
    output_folder = os.path.join("data", "video_segments")

    # Chop the video
    chop_video(input_path, output_folder, segment_duration=5, num_segments=3)

    print(f"Video chopping complete. Segments saved in {output_folder}")