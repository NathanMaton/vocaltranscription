import librosa
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from datetime import datetime

def analyze_song(audio_path, max_sections=30):
    y, sr = librosa.load(audio_path)
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    
    # Compute spectral contrast
    contrast = librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr)
    
    # Detect vocal activity
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    vocal_activity = np.mean(mfcc[1:5], axis=0)  # Focus on lower MFCCs for vocal detection
    
    # Detect onsets with lower threshold for more sensitivity
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    
    # Use peak_pick directly to set the threshold for general onsets
    onset_frames = librosa.util.peak_pick(onset_env, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=10)
    
    # Use peak_pick directly for vocal onsets as well
    vocal_onset_frames = librosa.util.peak_pick(vocal_activity, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.5, wait=10)
    
    # Convert frames to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    vocal_onset_times = librosa.frames_to_time(vocal_onset_frames, sr=sr)
    
    # Combine onset detection with spectral contrast and vocal activity for more robust change points
    change_points = []
    for i in range(1, len(onset_times)):
        spectral_change = np.mean(np.abs(contrast[:, onset_frames[i]] - contrast[:, onset_frames[i-1]]))
        vocal_change = np.abs(vocal_activity[onset_frames[i]] - vocal_activity[onset_frames[i-1]])
        if (spectral_change > 0.6 and vocal_change > 0.6) or onset_times[i] in vocal_onset_times:
            change_points.append(onset_times[i])
    
    # If we have more change points than max_sections, use K-means to cluster
    if len(change_points) > max_sections:
        kmeans = KMeans(n_clusters=max_sections, random_state=42)
        clustered = kmeans.fit_predict(np.array(change_points).reshape(-1, 1))
        timestamps = np.array([np.array(change_points)[clustered == i].mean() for i in range(max_sections)])
    else:
        timestamps = np.array(change_points)
    
    timestamps.sort()
    
    # Detect tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return timestamps, tempo

def generate_vibes(timestamp, total_duration):
    position = timestamp / total_duration
    if position < 0.25:
        return "Intro vibes"
    elif position < 0.75:
        return "Main section vibes"
    else:
        return "Outro vibes"

def create_storyboard(audio_path, output_path):
    timestamps, tempo = analyze_song(audio_path)
    y, sr = librosa.load(audio_path)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    df = pd.DataFrame(columns=['Timestamp', 'Lyrics', 'Image Ideas', 'Vibes'])
    
    for timestamp in timestamps:
        new_row = pd.DataFrame({
            'Timestamp': [f"{timestamp:.2f}"],
            'Lyrics': [""],  # Placeholder for lyrics
            'Image Ideas': [""],
            'Vibes': [generate_vibes(timestamp, total_duration)]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    tempo_value = tempo[0] if isinstance(tempo, np.ndarray) else tempo
    new_row = pd.DataFrame({
        'Timestamp': ["Overall"],
        'Lyrics': [""],
        'Image Ideas': [""],
        'Vibes': [f"Tempo: {tempo_value:.2f} BPM"]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    
    df.to_excel(output_path, index=False)
    print(f"Storyboard saved to {output_path}")

if __name__ == "__main__":
    audio_filename = "loommain.mp3"
    input_path = os.path.join("data", "audio_for_mvs", audio_filename)
    song_title = "Loom"
    
    # Generate timestamp for the file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_folder = os.path.join("data", "storyboards")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{song_title}_storyboard_{timestamp}.xlsx")
    
    create_storyboard(input_path, output_path)