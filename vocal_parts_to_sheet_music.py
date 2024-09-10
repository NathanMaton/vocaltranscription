import sys
import os
import numpy as np
from basic_pitch.inference import predict as bp_predict
import librosa
import pretty_midi
import music21 as m21
from scipy.cluster.vq import kmeans, vq

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

def detect_key_and_meter(midi_data):
    stream = m21.stream.Stream()
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            m21_note = m21.note.Note(note.pitch)
            m21_note.quarterLength = note.end - note.start
            stream.append(m21_note)
    
    key = stream.analyze('key')
    print(f"Detected key: {key.tonic.name} {key.mode}", file=sys.stderr)
    
    # Simple time signature detection
    total_duration = sum(n.quarterLength for n in stream.notesAndRests)
    num_measures = round(total_duration / 4)  # Assume 4 beats per measure
    if num_measures > 0:
        beats_per_measure = round(total_duration / num_measures)
        time_signature = m21.meter.TimeSignature(f'{beats_per_measure}/4')
    else:
        time_signature = m21.meter.TimeSignature('4/4')  # Default to 4/4 if detection fails
    
    print(f"Detected time signature: {time_signature.ratioString}", file=sys.stderr)
    return key, time_signature

def adaptive_quantize(duration):
    grid = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    return min(grid, key=lambda x: abs(x - duration))

def get_beat_positions(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    return beat_times, float(tempo)

def quantize_to_nearest_beat(note_time, beat_times):
    return beat_times[np.argmin(np.abs(np.array(beat_times) - note_time))]

def cluster_note_durations(durations, n_clusters=5):
    durations = np.array(durations).reshape(-1, 1)
    centroids, _ = kmeans(durations, n_clusters)
    return centroids.flatten().tolist()

def advanced_quantize(note_start, note_end, beat_times, duration_clusters):
    quantized_start = quantize_to_nearest_beat(note_start, beat_times)
    raw_duration = note_end - note_start
    quantized_duration = min(duration_clusters, key=lambda x: abs(x - raw_duration))
    
    # Round to nearest expressible duration, favoring longer durations
    expressible_durations = [1/4, 1/2, 1, 2, 4, 8]  # Removed shorter durations
    quantized_duration = min(expressible_durations, key=lambda x: abs(x - quantized_duration))
    
    return quantized_start, quantized_duration

def merge_short_notes(notes, minimum_duration=0.25):  # minimum_duration is a quarter note
    merged_notes = []
    current_note = None
    
    for note in notes:
        if current_note is None:
            current_note = note
        elif note.pitch == current_note.pitch and note.start - current_note.end < 0.1:  # Allow small gaps
            current_note.end = note.end
        else:
            if current_note.end - current_note.start >= minimum_duration:
                merged_notes.append(current_note)
            current_note = note
    
    if current_note and current_note.end - current_note.start >= minimum_duration:
        merged_notes.append(current_note)
    
    return merged_notes

def create_part_from_midi(midi_data, part_name, beat_times, tempo, detect_silence=False, key=None):
    part = m21.stream.Part()
    part.partName = part_name

    # Cluster note durations
    note_durations = [note.end - note.start for instrument in midi_data.instruments for note in instrument.notes]
    duration_clusters = cluster_note_durations(note_durations)

    # Convert beat_times to a list if it's a numpy array
    beat_times = beat_times.tolist() if isinstance(beat_times, np.ndarray) else beat_times

    # Ensure tempo is a scalar value
    tempo = float(tempo) if isinstance(tempo, np.ndarray) else tempo

    for instrument in midi_data.instruments:
        # Merge short notes before quantization
        merged_notes = merge_short_notes(instrument.notes)
        
        for note in merged_notes:
            note_start, note_duration = advanced_quantize(note.start, note.end, beat_times, duration_clusters)
            
            m21_note = m21.note.Note(note.pitch)
            m21_note.quarterLength = note_duration
            m21_note.volume.velocity = note.velocity

            if key:
                scale = key.getScale()
                scale_pitches = [p.name for p in scale.getPitches()]
                pitch_class = m21_note.pitch.pitchClass
                pitch_name = scale_pitches[pitch_class % len(scale_pitches)]
                m21_note.pitch.name = pitch_name

            part.append(m21_note)

    return part

def create_sheet_music(lead_midi, harmony_midi, lead_audio_path, harmony_audio_path, output_path, suffix, include_harmony=False):
    score = m21.stream.Score()

    try:
        detected_key, detected_time_signature = detect_key_and_meter(lead_midi)
        print(f"Detected key and time signature: {detected_key}, {detected_time_signature}")
    except Exception as e:
        print(f"Error in detect_key_and_meter: {str(e)}")
        return

    try:
        lead_beat_times, lead_tempo = get_beat_positions(lead_audio_path)
        print(f"Lead beat times and tempo detected: {len(lead_beat_times)} beats, tempo: {lead_tempo}")
    except Exception as e:
        print(f"Error in get_beat_positions for lead: {str(e)}")
        return

    try:
        lead_part = create_part_from_midi(lead_midi, "Lead Vocal", lead_beat_times, lead_tempo, detect_silence=False, key=detected_key)
        print("Lead part created successfully")
    except Exception as e:
        print(f"Error in create_part_from_midi for lead: {str(e)}")
        return

    lead_part.insert(0, m21.instrument.Instrument())
    lead_part.insert(0, detected_time_signature)
    key_signature = m21.key.KeySignature(detected_key.sharps)
    lead_part.insert(0, key_signature)
    
    try:
        lead_part.makeMeasures(inPlace=True)
        print("Measures created for lead part")
    except Exception as e:
        print(f"Error in makeMeasures for lead part: {str(e)}")
        return
    
    score.append(lead_part)
    
    if include_harmony and harmony_midi and harmony_audio_path:
        try:
            harmony_beat_times, harmony_tempo = get_beat_positions(harmony_audio_path)
            print(f"Harmony beat times and tempo detected: {len(harmony_beat_times)} beats, tempo: {harmony_tempo}")
        except Exception as e:
            print(f"Error in get_beat_positions for harmony: {str(e)}")
            return

        try:
            harmony_part = create_part_from_midi(harmony_midi, "Harmony Vocal", harmony_beat_times, harmony_tempo, detect_silence=True, key=detected_key)
            print("Harmony part created successfully")
        except Exception as e:
            print(f"Error in create_part_from_midi for harmony: {str(e)}")
            return

        harmony_part.insert(0, m21.instrument.Instrument())
        harmony_part.insert(0, detected_time_signature)
        harmony_part.insert(0, key_signature)
        
        lead_measures = len(lead_part.getElementsByClass('Measure'))
        try:
            harmony_part.makeMeasures(inPlace=True)
            print("Measures created for harmony part")
        except Exception as e:
            print(f"Error in makeMeasures for harmony part: {str(e)}")
            return

        harmony_measures = len(harmony_part.getElementsByClass('Measure'))
        
        if harmony_measures < lead_measures:
            for _ in range(lead_measures - harmony_measures):
                harmony_part.append(m21.stream.Measure())
        
        score.append(harmony_part)
    
    try:
        score.makeNotation(inPlace=True)
        print("Notation created for score")
    except Exception as e:
        print(f"Error in makeNotation: {str(e)}")
        return
    
    output_path_with_suffix = output_path.replace('.xml', f'_{suffix}.xml')
    try:
        score.write('musicxml', output_path_with_suffix)
        print(f"Sheet music created: {output_path_with_suffix}")
    except Exception as e:
        print(f"Error writing musicxml: {str(e)}")

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
            create_sheet_music(lead_midi, None, lead_path, None, output_path, "lead")
            
            if harmony_midi:
                create_sheet_music(harmony_midi, None, harmony_path, None, output_path, "harmony", include_harmony=True)
                create_sheet_music(lead_midi, harmony_midi, lead_path, harmony_path, output_path, "lead_and_harmony", include_harmony=True)
            else:
                print("Harmony vocal file not found. Skipping harmony and combined outputs.")
            
        except Exception as e:
            print(f"Error creating sheet music: {str(e)}")
    else:
        print("Failed to create sheet music due to errors in MIDI data extraction.")

if __name__ == "__main__":
    main()