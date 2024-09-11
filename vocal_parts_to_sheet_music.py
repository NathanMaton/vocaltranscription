import sys
import os
import numpy as np
from basic_pitch.inference import predict as bp_predict
import librosa
import pretty_midi
import music21 as m21
import traceback
import time
from tqdm import tqdm
import soundfile as sf
import pickle
from collections import Counter
import io
import tempfile

def preprocess_audio(audio_path, skip_noise_reduction=True):
    try:
        highpass_path = audio_path.replace('.wav', '_highpass.wav')
        if os.path.exists(highpass_path):
            print(f"Using existing high-pass filtered audio: {highpass_path}", file=sys.stderr)
            y, sr = librosa.load(highpass_path)
        else:
            print(f"Attempting to load audio file: {audio_path}", file=sys.stderr)
            y, sr = librosa.load(audio_path)
            print(f"Audio file loaded successfully. Duration: {len(y)/sr:.2f} seconds", file=sys.stderr)
            
            print("Applying high-pass filter...", file=sys.stderr)
            y_highpass = librosa.effects.hpss(y)[0]
            
            # Save the high-pass filtered audio
            sf.write(highpass_path, y_highpass, sr)
            print(f"High-pass filtered audio saved to: {highpass_path}", file=sys.stderr)
            y = y_highpass

        if not skip_noise_reduction:
            print("Applying noise reduction (this may take a while)...", file=sys.stderr)
            start_time = time.time()
            
            # Use a simpler noise reduction method
            y = librosa.decompose.nn_filter(y, aggregate=np.median, metric='cosine', width=int(sr/2))
            
            end_time = time.time()
            print(f"Noise reduction completed in {end_time - start_time:.2f} seconds", file=sys.stderr)
        else:
            print("Skipping noise reduction step.", file=sys.stderr)
        
        print("Preprocessing completed successfully.", file=sys.stderr)
        return y, sr
    except Exception as e:
        print(f"Error in preprocess_audio: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise

def examine_audio_and_prediction(audio_path, skip_noise_reduction=False, 
                                 onset_threshold=0.5, frame_threshold=0.3,
                                 minimum_note_length=0.058, 
                                 minimum_frequency=65, maximum_frequency=2093,
                                 multiple_pitch_bends=False, melodia_trick=True,
                                 merge_max_gap=0.15, merge_min_duration=0.075, merge_pitch_tolerance=1):
    try:
        y, sr = preprocess_audio(audio_path, skip_noise_reduction)
    except Exception as e:
        print(f"Error in preprocess_audio: {str(e)}", file=sys.stderr)
        return None

    try:
        # Create a filename for the cached Basic Pitch output
        cache_filename = f"{audio_path}_basic_pitch_output.pkl"
        
        if os.path.exists(cache_filename):
            print(f"Loading cached Basic Pitch output from {cache_filename}", file=sys.stderr)
            with open(cache_filename, 'rb') as f:
                model_output = pickle.load(f)
        else:
            # Create a temporary file for the preprocessed audio
            temp_path = audio_path.replace('.wav', '_preprocessed.wav')
            print(f"Writing preprocessed audio to temporary file: {temp_path}", file=sys.stderr)
            sf.write(temp_path, y, sr)

            print("Running Basic Pitch prediction...", file=sys.stderr)
            model_output = bp_predict(temp_path,
                                      onset_threshold=onset_threshold,
                                      frame_threshold=frame_threshold,
                                      minimum_note_length=minimum_note_length,
                                      minimum_frequency=minimum_frequency,
                                      maximum_frequency=maximum_frequency,
                                      multiple_pitch_bends=multiple_pitch_bends,
                                      melodia_trick=melodia_trick)
            
            # Save the model output
            with open(cache_filename, 'wb') as f:
                pickle.dump(model_output, f)
            print(f"Basic Pitch output saved to {cache_filename}", file=sys.stderr)
            
            # Remove the temporary file
            os.remove(temp_path)
            print("Temporary file removed.", file=sys.stderr)
        
        print(f"Type of model_output: {type(model_output)}", file=sys.stderr)
        
        if isinstance(model_output, tuple) and len(model_output) >= 2:
            midi_data = model_output[1]
            if isinstance(midi_data, pretty_midi.PrettyMIDI):
                print("MIDI data extracted successfully", file=sys.stderr)
                midi_data = merge_nearby_notes(midi_data, 
                                               max_gap=merge_max_gap, 
                                               min_duration=merge_min_duration, 
                                               pitch_tolerance=merge_pitch_tolerance)
                return midi_data
            else:
                print(f"Unexpected format for MIDI data: {type(midi_data)}", file=sys.stderr)
        else:
            print(f"Unexpected model output format: {type(model_output)}", file=sys.stderr)
        
        return None
    except Exception as e:
        print(f"Error in examine_audio_and_prediction: {str(e)}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None

def quantize_duration_16th(duration):
    if duration < 0.5:  # Less than 8th note
        return 0.25  # Minimum duration is 16th note
    elif duration < 1.0:  # Between 8th and quarter note
        return 0.5  # 8th note
    elif duration < 2.0:  # Between quarter and half note
        return 1.0  # Quarter note
    elif duration < 3.0:  # Between half and dotted half note
        return 2.0  # Half note
    else:  # Longer than dotted half note
        return round(duration / 1.0) * 1.0  # Round to nearest quarter note

def quantize_duration_8th(duration):
    if duration < 1.0:  # Less than quarter note
        return 0.5  # Minimum duration is 8th note
    elif duration < 2.0:  # Between quarter and half note
        return 1.0  # Quarter note
    elif duration < 3.0:  # Between half and dotted half note
        return 2.0  # Half note
    else:  # Longer than dotted half note
        return round(duration / 1.0) * 1.0  # Round to nearest quarter note

def quantize_duration_extended(duration):
    if duration < 0.75:  # Less than dotted 8th note
        return 0.5  # 8th note
    elif duration < 1.5:  # Between dotted 8th and dotted quarter
        return 1.0  # Quarter note
    elif duration < 2.5:  # Between dotted quarter and dotted half
        return 2.0  # Half note
    elif duration < 3.5:  # Between dotted half and dotted whole
        return 3.0  # Dotted half note
    else:  # Longer than dotted whole note
        return round(duration / 1.0) * 1.0  # Round to nearest quarter note

def create_part_from_midi(midi_data, part_name, quantize_func, detect_silence=False):
    part = m21.stream.Part()
    part.partName = part_name
    
    last_end_time = 0
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            if detect_silence:
                # Add a rest if there's a gap
                if note.start > last_end_time:
                    rest_duration = note.start - last_end_time
                    rest = m21.note.Rest()
                    rest.quarterLength = quantize_func(rest_duration)
                    if rest.quarterLength > 0:
                        part.append(rest)
            
            duration = note.end - note.start
            quantized_duration = quantize_func(duration)
            
            m21_note = m21.note.Note(note.pitch)
            m21_note.quarterLength = quantized_duration
            part.append(m21_note)
            
            last_end_time = note.end
    
    if detect_silence:
        # Add a final rest if needed
        if last_end_time < midi_data.get_end_time():
            final_rest_duration = midi_data.get_end_time() - last_end_time
            final_rest = m21.note.Rest()
            final_rest.quarterLength = quantize_func(final_rest_duration)
            if final_rest.quarterLength > 0:
                part.append(final_rest)
    
    return part

def create_sheet_music(lead_midi, harmony_midi, output_path, quantize_func, suffix, include_harmony=False, input_filename=''):
    score = m21.stream.Score()

    key = detect_key(lead_midi)
    
    lead_part = create_part_from_midi(lead_midi, "Lead Vocal", quantize_func, detect_silence=False)
    lead_part.insert(0, m21.instrument.Instrument())
    
    # Add time signature and detected key signature
    lead_part.insert(0, m21.meter.TimeSignature('4/4'))
    lead_part.insert(0, key)
    
    # Add measures
    lead_part.makeMeasures(inPlace=True)
    
    score.append(lead_part)
    
    if include_harmony and harmony_midi:
        harmony_part = create_part_from_midi(harmony_midi, "Harmony Vocal", quantize_func, detect_silence=True)
        harmony_part.insert(0, m21.instrument.Instrument())
        harmony_part.insert(0, m21.meter.TimeSignature('4/4'))
        harmony_part.insert(0, key)
        
        # Ensure harmony part has the same number of measures as lead part
        lead_measures = len(lead_part.getElementsByClass('Measure'))
        harmony_part.makeMeasures(inPlace=True)
        harmony_measures = len(harmony_part.getElementsByClass('Measure'))
        
        if harmony_measures < lead_measures:
            for _ in range(lead_measures - harmony_measures):
                harmony_part.append(m21.stream.Measure())
        
        score.append(harmony_part)
    
    # Clean up the score
    score.makeNotation(inPlace=True)
    
    # Write the score to a file or return as string
    if output_path == "memory":
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_file:
            score.write('musicxml', temp_file.name)
        
        with open(temp_file.name, 'r') as file:
            musicxml_content = file.read()
        
        os.unlink(temp_file.name)
        return musicxml_content
    else:
        output_filename = f"{os.path.splitext(input_filename)[0]}_{suffix}.xml"
        output_path_with_suffix = os.path.join(os.path.dirname(output_path), output_filename)
        score.write('musicxml', output_path_with_suffix)
        print(f"Sheet music created: {output_path_with_suffix}")

def test_configuration(lead_path, harmony_path, output_path, config, config_name):
    print(f"\nTesting configuration: {config_name}")
    lead_midi = examine_audio_and_prediction(lead_path, skip_noise_reduction=True, **config)
    harmony_midi = examine_audio_and_prediction(harmony_path, skip_noise_reduction=True, **config) if os.path.exists(harmony_path) else None
    
    if lead_midi:
        try:
            print("Key detection for lead vocal:")
            print_top_key_candidates(lead_midi)
            
            input_filename = os.path.basename(lead_path)
            
            # Lead vocal only, extended notes
            create_sheet_music(lead_midi, None, output_path, quantize_duration_extended, f"extended_lead_{config_name}", input_filename=input_filename)
            
            # Lead and harmony vocals, extended notes
            if harmony_midi:
                print("\nKey detection for harmony vocal:")
                print_top_key_candidates(harmony_midi)
                create_sheet_music(lead_midi, harmony_midi, output_path, quantize_duration_extended, f"extended_lead_and_harmony_{config_name}", include_harmony=True, input_filename=input_filename)
            
            print(f"Sheet music created for configuration {config_name}")
        except Exception as e:
            print(f"Error creating sheet music for configuration {config_name}: {str(e)}")
    else:
        print(f"Failed to create sheet music for configuration {config_name} due to errors in MIDI data extraction.")

def merge_nearby_notes(midi_data, max_gap=0.15, min_duration=0.075, pitch_tolerance=1):
    try:
        key = detect_key(midi_data)
        scale = key.getScale()
        scale_pitches = [note.midi % 12 for note in scale.getPitches()]
    except Exception as e:
        print(f"Error in key detection: {str(e)}")
        print("Proceeding without key-based filtering")
        scale_pitches = list(range(12))  # Consider all pitches as in-scale
    
    for instrument in midi_data.instruments:
        merged_notes = []
        current_note = None
        for note in instrument.notes:
            if current_note is None:
                current_note = note
            elif (abs(note.pitch - current_note.pitch) <= pitch_tolerance and 
                  note.start - current_note.end <= max_gap):
                # Merge notes if they're close in pitch and time
                current_note.end = max(current_note.end, note.end)
                current_note.velocity = max(current_note.velocity, note.velocity)
                # If the new note is in the scale and the current note isn't, update the pitch
                if note.pitch % 12 in scale_pitches and current_note.pitch % 12 not in scale_pitches:
                    current_note.pitch = note.pitch
            else:
                if current_note.end - current_note.start >= min_duration:
                    merged_notes.append(current_note)
                current_note = note
        if current_note is not None and current_note.end - current_note.start >= min_duration:
            merged_notes.append(current_note)
        
        # Filter out notes that are not in the key, unless they're longer than a threshold
        key_notes = []
        for note in merged_notes:
            if note.pitch % 12 in scale_pitches or note.end - note.start > 0.5:  # 0.5 seconds threshold
                key_notes.append(note)
        
        instrument.notes = key_notes
    return midi_data

def calculate_pitch_histogram(midi_data):
    pitch_hist = np.zeros(12)
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch_class = note.pitch % 12
            duration = note.end - note.start
            pitch_hist[pitch_class] += duration
    return pitch_hist / np.sum(pitch_hist)  # Normalize histogram

def detect_key(midi_data):
    # Krumhansl-Schmuckler key profiles
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    pitch_hist = calculate_pitch_histogram(midi_data)

    # Calculate correlation with each possible key
    correlations = []
    for i in range(12):
        major_corr = np.correlate(np.roll(pitch_hist, -i), major_profile)
        minor_corr = np.correlate(np.roll(pitch_hist, -i), minor_profile)
        correlations.append((major_corr[0], 'major'))
        correlations.append((minor_corr[0], 'minor'))

    # Find the key with the highest correlation
    best_key = max(correlations, key=lambda x: x[0])
    tonic = correlations.index(best_key) // 2
    mode = best_key[1]

    tonic_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][tonic]
    print(f"Detected key: {tonic_name} {mode}")

    # Create a music21 key object
    return m21.key.Key(tonic_name, mode)

def print_top_key_candidates(midi_data):
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    pitch_hist = calculate_pitch_histogram(midi_data)

    correlations = []
    for i in range(12):
        major_corr = np.correlate(np.roll(pitch_hist, -i), major_profile)
        minor_corr = np.correlate(np.roll(pitch_hist, -i), minor_profile)
        correlations.append((major_corr[0], i, 'major'))
        correlations.append((minor_corr[0], i, 'minor'))

    # Sort correlations in descending order
    sorted_correlations = sorted(correlations, key=lambda x: x[0], reverse=True)

    print("Top 3 key candidates:")
    for i in range(3):
        corr, tonic, mode = sorted_correlations[i]
        tonic_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][tonic]
        print(f"{i+1}. {tonic_name} {mode} (correlation: {corr:.4f})")

def main():
    output_dir = "data/extracted_audio"
    lead_path = os.path.join(output_dir, "billyjeanlead.wav")
    harmony_path = os.path.join(output_dir, "billyjeanharmony.wav")
    output_path = "data/sheet_music/vocal_sheet_music.xml"
    
    if not os.path.exists(lead_path):
        print("Error: Lead vocal file not found. Please run extract_audio.py first.")
        return
    
    if not os.path.exists(harmony_path):
        print("Warning: Harmony vocal file not found. Only lead vocal will be processed.")
    
    # Define a single configuration with intermediate merging values
    configurations = [
        {
            "onset_threshold": 0.5,
            "frame_threshold": 0.3,
            "minimum_note_length": 0.058,
            "minimum_frequency": 65,
            "maximum_frequency": 2093,
            "multiple_pitch_bends": False,
            "melodia_trick": True,
            "merge_max_gap": 0.15,  # Intermediate value between 0.1 and 0.2
            "merge_min_duration": 0.075,  # Intermediate value between 0.05 and 0.1
            "merge_pitch_tolerance": 1  # Keep at 1 for more precise pitch matching
        }
    ]
    
    for i, config in enumerate(configurations, 1):
        test_configuration(lead_path, harmony_path, output_path, config, f"balanced_merge")
    
if __name__ == "__main__":
    main()