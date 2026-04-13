from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
import os
import random
import torch

from flask import Flask, jsonify, request, send_file, after_this_request
from TTS.api import TTS

APP_DIR = Path(__file__).resolve().parent

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_DIR = APP_DIR / "voices"
INDEX_FILE = APP_DIR / "index.html"

ALLOWED_LANGUAGES = {
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
    "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
}

VOICE_SUFFIXES = {".wav"}
tts_lock = Lock()

app = Flask(__name__)

if not VOICE_DIR.is_dir():
    raise FileNotFoundError(f"Voice directory not found: {VOICE_DIR}")

def get_reference_wavs() -> list[Path]:
    wavs  = sorted(
        p for p in VOICE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in VOICE_SUFFIXES
    )
    if not wavs:
        raise FileNotFoundError(f"No reference WAV files found in: {VOICE_DIR}")
    return wavs

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

tts = TTS(MODEL_NAME).to(device)

def choose_reference_wav() -> Path:
    return random.choice(get_reference_wavs())

@app.get("/")
def index():
    return send_file(INDEX_FILE)

@app.get("/health")
def health():
    try:
        reference_wavs = get_reference_wavs()
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({
        "ok": True,
        "model": MODEL_NAME,
        "voice_count": len(reference_wavs),
        "voices": [p.name for p in reference_wavs],
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
    print(f" > Reference WAV: {reference_wav.name}")

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
