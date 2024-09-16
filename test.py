import sys
import os
import numpy as np
import soundfile as sf
import tensorflow as tf
import crepe
from music21 import note, stream, duration

print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CREPE version: {crepe.__version__}")

# Step 1: Load the audio file
audio_path = 'data/extracted_audio/knowbetterleadvocal.wav'  # Replace with your .wav file path
audio, sr = sf.read(audio_path)
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # Convert to mono if stereo

# Resample to 16000 Hz if necessary
if sr != 16000:
    from scipy import signal
    audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    sr = 16000

# Step 2: Use CREPE to estimate pitch
time_steps, frequencies, confidences, activation = crepe.predict(audio, sr, viterbi=True)

# Optional: Filter out low-confidence estimates
confidence_threshold = 0.5  # Adjust as needed
frequencies = np.where(confidences >= confidence_threshold, frequencies, 0)

# Step 3: Convert frequencies to MIDI note numbers
def frequency_to_midi(frequency):
    return 69 + 12 * np.log2(frequency / 440.0)

midi_notes = frequency_to_midi(frequencies)
midi_notes = np.where(frequencies > 0, midi_notes, 0)

# Round MIDI notes to the nearest integer
midi_notes = np.round(midi_notes).astype(int)

# Step 4: Create a music21 stream
s = stream.Stream()

# Initialize variables
prev_midi_note = None
note_start_time = None

# Iterate over the estimated MIDI notes
for i, midi_note in enumerate(midi_notes):
    current_time = time_steps[i]
    if midi_note > 0:
        if prev_midi_note is None:
            # Start a new note
            n = note.Note()
            n.pitch.midi = midi_note
            note_start_time = current_time
            prev_midi_note = midi_note
        elif midi_note != prev_midi_note:
            # End the previous note and start a new one
            n.duration = duration.Duration(current_time - note_start_time)
            s.append(n)
            n = note.Note()
            n.pitch.midi = midi_note
            note_start_time = current_time
            prev_midi_note = midi_note
    else:
        if prev_midi_note is not None:
            # End the previous note
            n.duration = duration.Duration(current_time - note_start_time)
            s.append(n)
            prev_midi_note = None

# Add the last note if it hasn't been added
if prev_midi_note is not None:
    n.duration = duration.Duration(time_steps[-1] - note_start_time)
    s.append(n)

# Step 5: Export to MIDI file
midi_file_path = 'transcribed_vocal.mid'
s.write('midi', fp=midi_file_path)
print(f"Transcription saved as {midi_file_path}")

# Optional: Show the sheet music
s.show()
