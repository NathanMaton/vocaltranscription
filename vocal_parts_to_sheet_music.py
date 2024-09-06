import sys
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

def quantize_duration(duration):
    if duration < 0.25:  # Less than 16th note
        return 0  # Remove very short notes
    elif duration < 0.75:  # Between 16th and 8th note
        return 0.5  # 8th note
    elif duration < 1.5:  # Between 8th and quarter note
        return 1.0  # Quarter note
    elif duration < 2.5:  # Between quarter and half note
        return 2.0  # Half note
    else:  # Longer than half note
        return round(duration / 1.0) * 1.0  # Round to nearest quarter note

def filter_and_smooth_notes(notes, min_duration=0.1, pitch_threshold=2):
    filtered_notes = []
    for note in notes:
        if note.end - note.start >= min_duration:
            if filtered_notes and abs(note.pitch - filtered_notes[-1].pitch) <= pitch_threshold:
                # Merge with previous note if pitch is similar
                filtered_notes[-1].end = note.end
            else:
                filtered_notes.append(note)
    return filtered_notes

def create_part_from_midi(midi_data, part_name):
    part = m21.stream.Part()
    part.partName = part_name
    
    last_end_time = 0
    for instrument in midi_data.instruments:
        notes = filter_and_smooth_notes(instrument.notes)
        for note in notes:
            # Add a rest if there's a gap
            if note.start > last_end_time:
                rest_duration = note.start - last_end_time
                rest = m21.note.Rest()
                rest.quarterLength = quantize_duration(rest_duration)
                if rest.quarterLength > 0:
                    part.append(rest)
            
            duration = note.end - note.start
            quantized_duration = quantize_duration(duration)
            
            if quantized_duration > 0:
                m21_note = m21.note.Note(note.pitch)
                m21_note.quarterLength = quantized_duration
                part.append(m21_note)
            
            last_end_time = note.end
    
    # Add a final rest if needed
    if last_end_time < midi_data.get_end_time():
        final_rest_duration = midi_data.get_end_time() - last_end_time
        final_rest = m21.note.Rest()
        final_rest.quarterLength = quantize_duration(final_rest_duration)
        if final_rest.quarterLength > 0:
            part.append(final_rest)
    
    return part

def create_sheet_music(lead_midi, harmony_midi, output_path):
    score = m21.stream.Score()
    
    lead_part = create_part_from_midi(lead_midi, "Lead Vocal")
    harmony_part = create_part_from_midi(harmony_midi, "Harmony Vocal")
    
    # Add time signature and key signature to both parts
    for part in [lead_part, harmony_part]:
        part.insert(0, m21.meter.TimeSignature('4/4'))
        part.insert(0, m21.key.Key('C'))
    
    # Add measures
    lead_part.makeMeasures(inPlace=True)
    harmony_part.makeMeasures(inPlace=True)
    
    score.append(lead_part)
    score.append(harmony_part)
    
    # Write the score to a file
    score.write('musicxml', output_path)

def main():
    lead_path = "data/extracted_audio/knowbetterleadvocal.wav"
    harmony_path = "data/extracted_audio/knowbetterharmonyvocal.wav"
    output_path = "data/sheet_music/combined_vocal_sheet_music.xml"
    
    lead_midi = examine_audio_and_prediction(lead_path)
    harmony_midi = examine_audio_and_prediction(harmony_path)
    
    if lead_midi and harmony_midi:
        try:
            create_sheet_music(lead_midi, harmony_midi, output_path)
            print(f"Combined sheet music created: {output_path}")
        except Exception as e:
            print(f"Error creating sheet music: {str(e)}")
    else:
        print("Failed to create sheet music due to errors in MIDI data extraction.")

if __name__ == "__main__":
    main()