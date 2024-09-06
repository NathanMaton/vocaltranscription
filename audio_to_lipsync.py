import librosa
import numpy as np
from scipy.signal import resample
import json
import os
import sys
import matplotlib.pyplot as plt

def extract_audio_features(audio_path, n_mfcc=13):
    # Load the audio file (MP3 or WAV)
    y, sr = librosa.load(audio_path)
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Normalize MFCC to make it invariant to audio volume
    mfcc_normalized = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.std(mfcc, axis=1, keepdims=True)
    
    return mfcc_normalized, sr

def generate_lipsync_data(mfcc, sr, fps=30):
    # Resample MFCC to match desired frame rate
    target_length = int(mfcc.shape[1] * fps / sr)
    mfcc_resampled = resample(mfcc, target_length, axis=1)
    
    # Simple mapping of MFCC to lip positions
    lip_positions = np.mean(mfcc_resampled[:4], axis=0)
    
    # Normalize lip positions to range [0, 1]
    lip_positions = (lip_positions - np.min(lip_positions)) / (np.max(lip_positions) - np.min(lip_positions))
    
    return lip_positions

def save_lipsync_data(lip_positions, output_path):
    # Create a list of frame data
    frames = [{"frame": i, "lip_position": float(pos)} for i, pos in enumerate(lip_positions)]
    
    # Save to JSON file for easy integration with other systems
    with open(output_path, 'w') as f:
        json.dump(frames, f)

def visualize_lipsync(lip_positions, output_path):
    frames = range(len(lip_positions))
    plt.figure(figsize=(12, 6))
    plt.plot(frames, lip_positions)
    plt.title('Lip-sync Visualization')
    plt.xlabel('Frame')
    plt.ylabel('Lip Position')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Hardcode the audio filename
    audio_filename = "15seclakes.mp3"

    input_path = os.path.join("data", "audio_for_mvs", audio_filename)
    
    # Check if the file exists
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)
    
    # Set output path
    output_folder = os.path.join("data", "lipsync")
    os.makedirs(output_folder, exist_ok=True)
    base_filename = os.path.splitext(audio_filename)[0]
    json_output_path = os.path.join(output_folder, f"{base_filename}_lipsync.json")
    viz_output_path = os.path.join(output_folder, f"{base_filename}_lipsync_viz.png")
    
    # Extract audio features
    mfcc, sr = extract_audio_features(input_path)
    
    # Generate lip-sync data
    lip_positions = generate_lipsync_data(mfcc, sr)
    
    # Save lip-sync data
    save_lipsync_data(lip_positions, json_output_path)
    
    # Visualize lip-sync data
    visualize_lipsync(lip_positions, viz_output_path)
    
    print(f"Lip-sync data saved to {json_output_path}")
    print(f"Lip-sync visualization saved to {viz_output_path}")