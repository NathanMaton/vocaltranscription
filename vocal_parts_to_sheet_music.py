import sys
import os
import numpy as np
from basic_pitch.inference import predict as bp_predict
import librosa
import pretty_midi
import music21 as m21

def examine_audio_and_prediction(audio_path):
    try:
        y, sr = librosa.load(audio_path)
        print(f"Audio file loaded successfully. Duration: {len(y)/sr:.2f} seconds", file=sys.stderr)
    except Exception as e:
        print(f"Error loading audio file: {str(e)}", file=sys.stderr)
        return None

    model_output = bp_predict(audio_path)
    print(f"Type of model_output: {type(model_output)}", file=sys.stderr)
    
    if isinstance(model_output, tuple) and len(model_output) >= 2:
        midi_data = model_output[1]
        if isinstance(midi_data, pretty_midi.PrettyMIDI):
            print("MIDI data extracted successfully", file=sys.stderr)
            return midi_data
        else:
            print("Unexpected format for MIDI data", file=sys.stderr)
    else:
        print("Unexpected model output format", file=sys.stderr)
    
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

def quantize_duration_quarter(duration):
    if duration < 2.0:  # Less than half note
        return 1.0  # Minimum duration is quarter note
    elif duration < 3.0:  # Between half and dotted half note
        return 2.0  # Half note
    else:  # Longer than dotted half note
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

def main():
    output_dir = "data/extracted_audio"
    
    lead_path = os.path.join(output_dir, "knowbetterleadvocal.wav")
    harmony_path = os.path.join(output_dir, "knowbetterharmonyvocal.wav")
    output_path = "data/sheet_music/vocal_sheet_music.xml"
    
    if not os.path.exists(lead_path):
        print("Error: Lead vocal file not found. Please run extract_audio.py first.")
        return
    
    lead_midi = examine_audio_and_prediction(lead_path)
    harmony_midi = examine_audio_and_prediction(harmony_path) if os.path.exists(harmony_path) else None
    
    if lead_midi:
        try:
            # Lead vocal only, 16th notes
            create_sheet_music(lead_midi, None, output_path, quantize_duration_16th, "16th_lead")
            
            # Lead vocal only, 8th notes
            create_sheet_music(lead_midi, None, output_path, quantize_duration_8th, "8th_lead")
            
            # Lead vocal only, quarter notes
            create_sheet_music(lead_midi, None, output_path, quantize_duration_quarter, "quarter_lead")
            
            if harmony_midi:
                # Harmony vocal only, 8th notes
                create_sheet_music(harmony_midi, None, output_path, quantize_duration_8th, "8th_harmony", include_harmony=True)
                
                # Lead and harmony vocals, 8th notes
                create_sheet_music(lead_midi, harmony_midi, output_path, quantize_duration_8th, "8th_lead_and_harmony", include_harmony=True)
            else:
                print("Harmony vocal file not found. Skipping harmony and combined outputs.")
            
        except Exception as e:
            print(f"Error creating sheet music: {str(e)}")
    else:
        print("Failed to create sheet music due to errors in MIDI data extraction.")

if __name__ == "__main__":
    main()