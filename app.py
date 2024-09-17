from flask import Flask, render_template, request, jsonify, send_file
import os
import requests
import tempfile
from pydub import AudioSegment
from vocal_parts_to_sheet_music import examine_audio_and_prediction, create_sheet_music, quantize_duration_extended
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Add CSP headers
@app.after_request
def add_header(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://code.jquery.com https://cdnjs.cloudflare.com https://www.verovio.org; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' https:; connect-src 'self' https://itunes.apple.com https://gleitz.github.io; font-src 'self' https://gleitz.github.io;"
    return response

@app.route('/')
def index():
    disclaimer = "This tool is very much a test and does not yet accurately transcribe music. The purpose is to search songs you want to learn vocal melodies and harmonies for, then select an audio segment from them, then process the audio and get sheet music. Right now the sheet music you'll see given isn't very accurate."
    return render_template('index.html', disclaimer=disclaimer)

@app.route('/search', methods=['POST'])
def search_song():
    query = request.json['query']
    
    search_url = f'https://itunes.apple.com/search?term={query}&entity=song&limit=1'
    response = requests.get(search_url)
    
    if response.status_code == 200:
        results = response.json()
        if results['resultCount'] > 0:
            track = results['results'][0]
            return jsonify({
                'title': track['trackName'],
                'artist': track['artistName'],
                'url': track['previewUrl']
            })
    
    return jsonify({
        'error': 'No results found'
    }), 404

@app.route('/process', methods=['POST'])
def process_audio():
    app.logger.debug("Processing audio request received")
    try:
        data = request.json
        start_time = float(data.get('start_time', 0))
        end_time = float(data.get('end_time', 0))
        audio_url = data.get('audio_url')

        if not audio_url:
            app.logger.error("Audio URL is missing")
            return jsonify({'error': 'Audio URL is missing'}), 400

        app.logger.debug(f"Downloading audio from {audio_url}")
        response = requests.get(audio_url)
        if response.status_code != 200:
            app.logger.error(f"Failed to download audio. Status code: {response.status_code}")
            return jsonify({'error': 'Failed to download audio'}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix='.m4a') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        app.logger.debug(f"Audio saved to temporary file: {temp_file_path}")

        app.logger.debug("Converting audio to WAV and extracting selected portion")
        audio = AudioSegment.from_file(temp_file_path, format="m4a")
        selected_audio = audio[int(start_time*1000):int(end_time*1000)]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_file:
            selected_audio.export(wav_file.name, format="wav")
            wav_file_path = wav_file.name

        app.logger.debug(f"Selected audio saved to: {wav_file_path}")

        app.logger.debug("Processing audio and generating sheet music")
        processing_message = "Processing audio. This may take 30-60 seconds. Please wait..."
        lead_midi = examine_audio_and_prediction(wav_file_path, skip_noise_reduction=True)
        
        if lead_midi:
            musicxml = create_sheet_music(lead_midi, None, "memory", quantize_duration_extended, "processed", input_filename="processed.xml")
            
            # Generate MIDI file
            midi_buffer = io.BytesIO()
            lead_midi.write(midi_buffer)
            midi_buffer.seek(0)
            
            app.logger.debug("Sheet music and MIDI generated successfully")
            
            # Clean up temporary files
            os.unlink(temp_file_path)
            os.unlink(wav_file_path)
            
            return jsonify({
                'musicxml': musicxml,
                'midi': midi_buffer.getvalue().hex(),  # Send MIDI data as hexadecimal string
                'processing_message': processing_message
            })
        else:
            app.logger.error("Failed to process audio: No MIDI data generated")
            # Clean up temporary files
            os.unlink(temp_file_path)
            os.unlink(wav_file_path)
            return jsonify({'error': 'Failed to process audio: No MIDI data generated'}), 500
    except Exception as e:
        app.logger.exception("An error occurred during audio processing")
        return jsonify({'error': str(e)}), 500

@app.route('/midi/<path:filename>')
def serve_midi(filename):
    return send_file(f'static/midi/{filename}', mimetype='audio/midi')

@app.route('/static/soundfonts/<path:filename>')
def serve_soundfont(filename):
    app.logger.debug(f"Serving soundfont file: {filename}")
    return send_file(f'static/soundfonts/{filename}', mimetype='application/javascript')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))