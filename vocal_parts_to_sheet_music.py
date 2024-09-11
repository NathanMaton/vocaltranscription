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
                                 multiple_pitch_bends=False, melodia_trick=True):
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
                midi_data = merge_nearby_notes(midi_data)
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

def create_sheet_music(lead_midi, harmony_midi, output_path, quantize_func, suffix, include_harmony=False):
    score = m21.stream.Score()

    lead_part = create_part_from_midi(lead_midi, "Lead Vocal", quantize_func, detect_silence=False)
    lead_part.insert(0, m21.instrument.Instrument())
    
    # Add time signature and key signature
    lead_part.insert(0, m21.meter.TimeSignature('4/4'))
    lead_part.insert(0, m21.key.Key('C'))
    
    # Add measures
    lead_part.makeMeasures(inPlace=True)
    
    score.append(lead_part)
    
    if include_harmony and harmony_midi:
        harmony_part = create_part_from_midi(harmony_midi, "Harmony Vocal", quantize_func, detect_silence=True)
        harmony_part.insert(0, m21.instrument.Instrument())
        harmony_part.insert(0, m21.meter.TimeSignature('4/4'))
        harmony_part.insert(0, m21.key.Key('C'))
        
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
    
    # Write the score to a file
    output_path_with_suffix = output_path.replace('.xml', f'_{suffix}.xml')
    score.write('musicxml', output_path_with_suffix)
    print(f"Sheet music created: {output_path_with_suffix}")

def test_configuration(lead_path, harmony_path, output_path, config, config_name):
    print(f"\nTesting configuration: {config_name}")
    lead_midi = examine_audio_and_prediction(lead_path, skip_noise_reduction=True, **config)
    harmony_midi = examine_audio_and_prediction(harmony_path, skip_noise_reduction=True, **config) if os.path.exists(harmony_path) else None
    
    if lead_midi:
        try:
            # Lead vocal only, extended notes
            create_sheet_music(lead_midi, None, output_path, quantize_duration_extended, f"extended_lead_{config_name}")
            
            # Lead and harmony vocals, extended notes
            if harmony_midi:
                create_sheet_music(lead_midi, harmony_midi, output_path, quantize_duration_extended, f"extended_lead_and_harmony_{config_name}", include_harmony=True)
            
            print(f"Sheet music created for configuration: {config_name}")
        except Exception as e:
            print(f"Error creating sheet music for configuration {config_name}: {str(e)}")
    else:
        print(f"Failed to create sheet music for configuration {config_name} due to errors in MIDI data extraction.")

def merge_nearby_notes(midi_data, max_gap=0.1, min_duration=0.2):
    for instrument in midi_data.instruments:
        merged_notes = []
        current_note = None
        for note in instrument.notes:
            if current_note is None:
                current_note = note
            elif note.pitch == current_note.pitch and note.start - current_note.end <= max_gap:
                current_note.end = note.end
            else:
                if current_note.end - current_note.start >= min_duration:
                    merged_notes.append(current_note)
                current_note = note
        if current_note is not None and current_note.end - current_note.start >= min_duration:
            merged_notes.append(current_note)
        instrument.notes = merged_notes
    return midi_data

def main():
    output_dir = "data/extracted_audio"
    lead_path = os.path.join(output_dir, "knowbetterleadvocal.wav")
    harmony_path = os.path.join(output_dir, "knowbetterharmonyvocal.wav")
    output_path = "data/sheet_music/vocal_sheet_music.xml"
    
    if not os.path.exists(lead_path):
        print("Error: Lead vocal file not found. Please run extract_audio.py first.")
        return
    
    if not os.path.exists(harmony_path):
        print("Warning: Harmony vocal file not found. Only lead vocal will be processed.")
    
    # Define a single configuration with default values
    configurations = [
        {
            "onset_threshold": 0.5,
            "frame_threshold": 0.3,
            "minimum_note_length": 0.1,  # Increased from 0.058
            "minimum_frequency": 65,
            "maximum_frequency": 2093,
            "multiple_pitch_bends": False,
            "melodia_trick": True
        }
    ]
    
    for i, config in enumerate(configurations, 1):
        test_configuration(lead_path, harmony_path, output_path, config, f"extended_notes")
    
if __name__ == "__main__":
    main()