from flask import Flask, abort, request
from tempfile import NamedTemporaryFile
import whisper
import torch

# Check if NVIDIA GPU is available
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model:
model = whisper.load_model("base", device=DEVICE)

app = Flask(__name__)


@app.route("/")
def hello():
    return "Whisper Hello World!"


@app.route('/whisper', methods=['POST'])
def handler():
    if not request.files:
        # If the user didn't submit any files, return a 400 (Bad Request) error.
        abort(400)

    # For each file, let's store the results in a list of dictionaries.
    results = []

    # Loop over every file that the user submitted.
    for filename, handle in request.files.items():
        # Create a temporary file.
        # The location of the temporary file is available in `temp.name`.
        temp = NamedTemporaryFile()
        # Write the user's uploaded file to the temporary file.
        # The file will get deleted when it drops out of scope.
        handle.save(temp)
        # Let's get the transcript of the temporary file.
        result = model.transcribe(temp.name, word_timestamps=True)

        transcript = []
        for segment in result['segments']:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            words = segment['words']
            transcript.append({
                'start': start_time,
                'end': end_time,
                'text': text,
                'words': words
            })

        results.append({
            'filename': filename,
            'text': result['text'],
            'transcript': transcript
        })

    # This will be automatically converted to JSON.
    return {'results': results}
