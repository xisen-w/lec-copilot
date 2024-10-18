import logging
from flask import Flask, render_template, request, jsonify
from Agents.lectureAgent import LectureAgent  # Import the LectureAgent
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the LectureAgent
lecture_agent = LectureAgent(model_name="gpt-4o-mini")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():
    if 'audio_file' not in request.files:
        app.logger.error("No audio file part in the request")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        app.logger.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    # Save the audio file securely
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join('/tmp', filename)
    audio_file.save(audio_path)

    try:
        # Use LectureAgent to transcribe and process the audio
        transcription = lecture_agent.record_lecture(audio_path)
        idea_cards = lecture_agent.explain(transcription)
        
        # Format the response
        response = {
            "transcription": transcription,
            "idea_cards": [{"idea_name": card.idea_name, "idea_explanation": card.idea_explanation} for card in idea_cards]
        }

        return jsonify(response)
    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)