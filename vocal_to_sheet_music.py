import librosa
import numpy as np
from music21 import stream, note, chord, pitch, meter, key, analysis
import time
from scipy import ndimage
import pickle
import os
from fractions import Fraction
import soundfile as sf  # Add this import
from midiutil import MIDIFile

def save_interim(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved interim file: {filename}")

def load_interim(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded interim file: {filename}")
        return data
    return None

def save_audio(y, sr, filename):
    sf.write(filename, y, sr)
    print(f"Saved audio file: {filename}")

def extract_vocals(audio_file, interim_dir):
    vocals_file = os.path.join(interim_dir, 'vocals.pkl')
    interim_data = load_interim(vocals_file)
    if interim_data:
        y_foreground, sr = interim_data
        save_audio(y_foreground, sr, os.path.join(interim_dir, 'vocals.wav'))
        return interim_data

    print("Loading audio file...")
    y, sr = librosa.load(audio_file)
    save_audio(y, sr, os.path.join(interim_dir, 'original.wav'))
    print(f"Audio file loaded. Sample rate: {sr}Hz, Duration: {len(y)/sr:.2f} seconds")
    
    print("Separating vocals (this may take a while)...")
    start_time = time.time()
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2
    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)
    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)
    S_foreground = mask_v * S_full
    y_foreground = librosa.istft(S_foreground * phase)
    print(f"Vocal separation completed in {time.time() - start_time:.2f} seconds")
    
    save_audio(y_foreground, sr, os.path.join(interim_dir, 'vocals.wav'))
    save_interim((y_foreground, sr), vocals_file)
    return y_foreground, sr

def extract_pitch(y, sr, interim_dir):
    pitch_file = os.path.join(interim_dir, 'pitch.pkl')
    interim_data = load_interim(pitch_file)
    if interim_data:
        f0, voiced_flag = interim_data
        synthesized = synthesize_pitch(f0, voiced_flag, sr)
        save_audio(synthesized, sr, os.path.join(interim_dir, 'pitch_synth.wav'))
        return interim_data

    print("Extracting pitch...")
    start_time = time.time()
    
    # Adjust these parameters
    frame_length = 1024
    hop_length = 256
    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')
    
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                 frame_length=frame_length,
                                                 hop_length=hop_length,
                                                 fmin=fmin,
                                                 fmax=fmax)
    
    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds")
    print(f"Number of frames: {len(f0)}")
    print(f"Number of voiced frames: {np.sum(voiced_flag)}")
    print(f"Min frequency: {np.min(f0[voiced_flag]):.2f} Hz")
    print(f"Max frequency: {np.max(f0[voiced_flag]):.2f} Hz")
    print(f"Shape of y: {y.shape}")
    print(f"Shape of f0: {f0.shape}")
    print(f"Shape of voiced_flag: {voiced_flag.shape}")
    
    synthesized = synthesize_pitch(f0, voiced_flag, sr)
    save_audio(synthesized, sr, os.path.join(interim_dir, 'pitch_synth.wav'))
    save_midi(f0, voiced_flag, os.path.join(interim_dir, 'pitch.mid'))
    save_interim((f0, voiced_flag), pitch_file)
    return f0, voiced_flag

def quantize_pitch(f0):
    midi_notes = librosa.hz_to_midi(f0)
    quantized_notes = np.round(midi_notes)
    return quantized_notes

def smooth_notes(notes, voiced_flag, min_duration=3):
    smoothed_notes = np.copy(notes)
    for i in range(1, len(notes) - 1):
        if voiced_flag[i] and voiced_flag[i-1] and voiced_flag[i+1]:
            smoothed_notes[i] = np.median(notes[i-1:i+2])
    
    voiced_int = voiced_flag.astype(int)
    mask = ndimage.median_filter(voiced_int, size=min_duration) > 0
    smoothed_notes[~mask] = 0
    
    return smoothed_notes

def quantize_duration(duration):
    # List of common note durations (in quarter notes)
    common_durations = [4, 2, 1, 0.5, 0.25, 0.125]
    return min(common_durations, key=lambda x: abs(x - duration))

def create_sheet_music(notes, voiced_flag, sr, output_file):
    print("Creating sheet music...")
    start_time = time.time()
    s = stream.Stream()
    
    pitch_notes = [note.Note(pitch.Pitch(midi=int(n))) for n in notes[voiced_flag] if n > 0]
    if pitch_notes:
        pitch_stream = stream.Stream(pitch_notes)
        k = analysis.discrete.analyzeStream(pitch_stream, 'key')
        if k:
            s.append(k)
        else:
            print("Warning: Key detection failed.")
    else:
        print("Warning: No valid pitches found for key detection.")
    
    s.append(meter.TimeSignature('4/4'))
    
    current_note = None
    current_duration = 0
    
    for note_value, is_voiced in zip(notes, voiced_flag):
        if is_voiced and note_value > 0:
            midi_value = int(round(note_value))
            if current_note is None or midi_value != current_note.pitch.midi:
                if current_note is not None:
                    duration_in_quarter_notes = current_duration / (sr / 4)
                    quantized_duration = quantize_duration(duration_in_quarter_notes)
                    current_note.quarterLength = quantized_duration
                    s.append(current_note)
                current_note = note.Note(midi_value)
                current_duration = 1
            else:
                current_duration += 1
        elif current_note is not None:
            duration_in_quarter_notes = current_duration / (sr / 4)
            quantized_duration = quantize_duration(duration_in_quarter_notes)
            current_note.quarterLength = quantized_duration
            s.append(current_note)
            current_note = None
            current_duration = 0
    
    if current_note is not None:
        duration_in_quarter_notes = current_duration / (sr / 4)
        quantized_duration = quantize_duration(duration_in_quarter_notes)
        current_note.quarterLength = quantized_duration
        s.append(current_note)
    
    print(f"Writing MusicXML file: {output_file}")
    s.write('musicxml', output_file)
    print(f"Sheet music creation completed in {time.time() - start_time:.2f} seconds")

def synthesize_pitch(f0, voiced_flag, sr):
    # Estimate hop length based on the length of f0 and the sample rate
    hop_length = len(voiced_flag) // len(f0)
    
    # Ensure hop_length is at least 1
    hop_length = max(1, hop_length)
    
    # Create time array
    t = np.arange(0, len(f0) * hop_length) / sr
    signal = np.zeros_like(t)
    
    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if voiced and freq > 0:
            start = i * hop_length
            end = (i + 1) * hop_length
            period = int(sr / freq)
            if period > 0:  # Ensure we don't divide by zero
                saw = np.linspace(1, -1, period)
                signal[start:end] = np.tile(saw, (end - start) // period + 1)[:end-start]
    
    # Apply a simple envelope to reduce clicks
    envelope = np.ones_like(signal)
    attack = int(0.005 * sr)
    release = int(0.005 * sr)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)
    
    return signal * envelope

def save_midi(f0, voiced_flag, filename):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, 120)
    
    current_note = None
    start_time = 0
    
    for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
        if voiced and freq > 0:
            midi_note = int(round(librosa.hz_to_midi(freq)))
            if current_note != midi_note:
                if current_note is not None:
                    midi.addNote(0, 0, current_note, start_time, i - start_time, 100)
                current_note = midi_note
                start_time = i
        elif current_note is not None:
            midi.addNote(0, 0, current_note, start_time, i - start_time, 100)
            current_note = None
    
    with open(filename, "wb") as output_file:
        midi.writeFile(output_file)
    print(f"Saved MIDI file: {filename}")

def process_audio_to_sheet_music(input_file, output_file, data_folder):
    interim_dir = os.path.join(data_folder, 'interim_files')
    os.makedirs(interim_dir, exist_ok=True)
    
    output_dir = os.path.join(data_folder, 'sheet_music')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    print(f"Processing {input_file} to create sheet music...")
    start_time = time.time()
    
    vocals, sr = extract_vocals(input_file, interim_dir)
    print(f"Vocals shape: {vocals.shape}, Sample rate: {sr}")
    
    f0, voiced_flag = extract_pitch(vocals, sr, interim_dir)
    print(f"f0 shape: {f0.shape}, voiced_flag shape: {voiced_flag.shape}")
    
    quantized_notes = quantize_pitch(f0)
    smoothed_notes = smooth_notes(quantized_notes, voiced_flag)
    create_sheet_music(smoothed_notes, voiced_flag, sr, output_path)
    
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

# Example usage
if __name__ == "__main__":
    input_file = os.path.join('data', 'extracted_audio', "The_Civil_Wars_-_If_I_Didnt_Know_Better.mp3")
    output_file = "knowbetter.musicxml"
    data_folder = 'data'
    process_audio_to_sheet_music(input_file, output_file, data_folder)