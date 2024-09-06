import librosa
import music21 as m21
import numpy as np

def transcribe_to_sheet_music(audio_path, output_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Example: Transcribe using librosa's pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    stream = m21.stream.Stream()
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            note = m21.note.Note()
            note.pitch.frequency = pitch
            stream.append(note)
    
    stream.write('musicxml', output_path)

def main():
    input_path = "data/extracted_audio/leadvocal.wav"
    output_path = "data/sheet_music/leadvocal_sheet_music.xml"
    
    transcribe_to_sheet_music(input_path, output_path)
    print(f"Sheet music generated: {output_path}")

if __name__ == "__main__":
    main()