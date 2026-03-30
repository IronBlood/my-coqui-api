from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
import os
import random

from flask import Flask, jsonify, request, send_file, after_this_request
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_DIR = Path(__file__).with_name("voices")

ALLOWED_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
}

VOICE_SUFFIXES = {".wav"}
tts_lock = Lock()

app = Flask(__name__)

if not VOICE_DIR.is_dir():
    raise FileNotFoundError(f"Voice directory not found: {VOICE_DIR}")

REFERENCE_WAVS = sorted(
    p for p in VOICE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in VOICE_SUFFIXES
)

if not REFERENCE_WAVS:
    raise FileNotFoundError(f"No reference WAV files found in: {VOICE_DIR}")

tts = TTS(MODEL_NAME)

def choose_reference_wav() -> Path:
    return random.choice(REFERENCE_WAVS)

@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": MODEL_NAME,
        "voice_count": len(REFERENCE_WAVS),
        "voices": [p.name for p in REFERENCE_WAVS],
    })

@app.post("/tts")
def synthesize():
    text = (request.form.get("text") or "").strip()
    language = (request.form.get("language") or "").strip().lower()

    if not text:
        return jsonify({"error": "text is required"}), 400

    if language == "zh":
        language = "zh-cn"

    if language not in ALLOWED_LANGUAGES:
        return jsonify({"error": f"unsupported language: {language}"}), 400


    out_path = None
    reference_wav = choose_reference_wav()

    try:
        with tts_lock:
            with NamedTemporaryFile(delete=False, suffix=".wav") as out_file:
                out_path = out_file.name
            tts.tts_to_file(
                text=text,
                speaker_wav=str(reference_wav),
                language=language,
                file_path=out_path,
            )

        @after_this_request
        def cleanup(response):
            if out_path and os.path.exists(out_path):
                os.unlink(out_path)
            return response

        return send_file(
            out_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="output.wav",
        )
    except Exception as e:
        if out_path and os.path.exists(out_path):
            os.unlink(out_path)
        return jsonify({
            "error": str(e),
            "reference_wav": reference_wav.name,
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
