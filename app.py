import os
import re
import threading
import uuid
import wave
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from flask import Flask, jsonify, render_template, request, send_from_directory
from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3gen import flow_matching as flow_matching_module
from chatterbox.models.t3 import t3 as t3_module
from tqdm import tqdm as base_tqdm
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = None
MODEL_READY = False
MODEL_ERROR = None
MODEL_LOCK = threading.Lock()
GENERATION_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}
MODEL_PROGRESS = threading.local()
MAX_CHUNK_CHARACTERS = 220
CHUNK_SILENCE_SECONDS = 0.16


def progress_aware_tqdm(stage):
    """Expose Chatterbox's internal inference steps to the active job."""
    def wrapped(iterable, *args, **kwargs):
        progress_bar = base_tqdm(iterable, *args, **kwargs)
        for item in progress_bar:
            callback = getattr(MODEL_PROGRESS, "callback", None)
            if callback:
                callback(stage, progress_bar.n, progress_bar.total or 1)
            yield item
    return wrapped


# Chatterbox imports tqdm directly, so replace those two module-local references.
t3_module.tqdm = progress_aware_tqdm("sampling")
flow_matching_module.tqdm = progress_aware_tqdm("decoding")


def load_model():
    global MODEL, MODEL_READY, MODEL_ERROR
    if MODEL is not None:
        return MODEL

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        try:
            MODEL = ChatterboxTTS.from_pretrained(device=DEVICE)
            MODEL_READY = True
            MODEL_ERROR = None
        except Exception as exc:
            MODEL_ERROR = str(exc)
            MODEL_READY = False
            raise

    return MODEL


def preload_model():
    try:
        load_model()
    except Exception:
        pass


def prepare_reference_audio(input_path, output_path):
    try:
        data, sample_rate = sf.read(str(input_path), dtype="float32")
    except Exception:
        return None

    if data.ndim > 1:
        data = data[:, 0]

    length = min(len(data), int(sample_rate * 3))
    trimmed = data[:length]

    sf.write(str(output_path), trimmed, sample_rate)
    return str(output_path)


def save_wav(output_path, wav, sample_rate):
    wav = to_mono_float(wav)
    peak = float(np.max(np.abs(wav))) if wav.size else 1.0
    if peak > 0:
        wav = wav / peak

    pcm = np.clip(wav, -1.0, 1.0)
    pcm = np.round(pcm * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def to_mono_float(wav):
    """Convert model output to a one-dimensional float waveform."""
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu()

    if isinstance(wav, torch.Tensor):
        if wav.dim() > 1:
            wav = wav.squeeze(0)
        wav = wav.float().numpy()
    else:
        wav = np.asarray(wav)

    if wav.ndim > 1:
        wav = wav[0]
    return wav.astype(np.float32)


def split_text_for_tts(text, max_characters=MAX_CHUNK_CHARACTERS):
    """Split long scripts at natural boundaries before model generation."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks = []
    current = ""

    for sentence in sentences:
        # A very long sentence is split on word boundaries as a fallback.
        words = sentence.split()
        pieces = []
        piece = ""
        for word in words:
            candidate = f"{piece} {word}".strip()
            if piece and len(candidate) > max_characters:
                pieces.append(piece)
                piece = word
            else:
                piece = candidate
        if piece:
            pieces.append(piece)

        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > max_characters:
                chunks.append(current)
                current = piece
            else:
                current = candidate

    if current:
        chunks.append(current)
    return chunks


def generate_long_form_audio(
    model, text, reference_audio_path, exaggeration, progress_callback=None, model_step_callback=None
):
    chunks = split_text_for_tts(text)
    rendered = []

    if progress_callback:
        progress_callback(8, 0, len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        kwargs = {"exaggeration": exaggeration}
        if reference_audio_path is not None:
            kwargs["audio_prompt_path"] = str(reference_audio_path)
        MODEL_PROGRESS.callback = (
            lambda stage, step, total: model_step_callback(
                stage, index, len(chunks), step, total
            ) if model_step_callback else None
        )
        try:
            rendered.append(to_mono_float(model.generate(chunk, **kwargs)))
        finally:
            MODEL_PROGRESS.callback = None
        if progress_callback:
            progress_callback(8 + (84 * index / len(chunks)), index, len(chunks))

    silence = np.zeros(int(model.sr * CHUNK_SILENCE_SECONDS), dtype=np.float32)
    segments = []
    for index, rendered_chunk in enumerate(rendered):
        segments.append(rendered_chunk)
        if index < len(rendered) - 1:
            segments.append(silence)
    wav = np.concatenate(segments)
    if progress_callback:
        progress_callback(94, len(chunks), len(chunks))
    return wav, len(chunks)


def render_job(job_id, text, reference_audio_path, exaggeration, output_path, filename):
    """Render independently of the browser request, which may disconnect."""
    try:
        with JOBS_LOCK:
            JOBS[job_id].update({"status": "generating", "progress": 3})

        def report_progress(progress, completed_chunks, total_chunks):
            with JOBS_LOCK:
                JOBS[job_id].update(
                    {
                        "progress": progress,
                        "completed_chunks": completed_chunks,
                        "total_chunks": total_chunks,
                    }
                )

        def report_model_step(stage, chunk_index, total_chunks, step, total_steps):
            # Sampling is most of the work; final decoding accounts for the rest.
            stage_fraction = min(step / total_steps, 1.0)
            if stage == "sampling":
                within_chunk = stage_fraction * 0.88
            else:
                within_chunk = 0.88 + stage_fraction * 0.10
            progress = 3 + 94 * ((chunk_index - 1 + within_chunk) / total_chunks)
            with JOBS_LOCK:
                JOBS[job_id].update(
                    {
                        "progress": progress,
                        "completed_chunks": chunk_index - 1,
                        "total_chunks": total_chunks,
                        "stage": "sampling" if stage == "sampling" else "decoding",
                        "step": step,
                        "total_steps": total_steps,
                    }
                )

        # Chatterbox uses a shared model and should only render one job at a time.
        with GENERATION_LOCK:
            model = load_model()
            wav, chunk_count = generate_long_form_audio(
                model,
                text,
                reference_audio_path,
                exaggeration,
                report_progress,
                report_model_step,
            )
            save_wav(output_path, wav, model.sr)

        if not output_path.exists() or output_path.stat().st_size <= 1000:
            raise RuntimeError("Generated audio was empty; please try a different voice or text.")

        result = {
            "status": "complete",
            "audio_url": f"/audio/{filename}",
            "chunk_count": chunk_count,
            "progress": 100,
        }
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}

    with JOBS_LOCK:
        JOBS[job_id].update(result)


def get_available_voices():
    return [
        {"id": "default", "name": "Default American English voice"},
        {"id": "neutral", "name": "Neutral American English"},
        {"id": "warm", "name": "Warm American English"},
        {"id": "dramatic", "name": "Dramatic American English"},
        {"id": "energetic", "name": "Energetic American English"},
        {"id": "calm", "name": "Calm American English"},
    ]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/voices")
def list_voices():
    return jsonify(get_available_voices())


@app.route("/api/status")
def status():
    if MODEL_READY:
        return jsonify({"ready": True, "message": "Chatterbox is ready."})
    if MODEL_ERROR:
        return jsonify({"ready": False, "message": MODEL_ERROR}), 500
    return jsonify({"ready": False, "message": "Loading Chatterbox model..."})


@app.route("/api/tts", methods=["POST"])
def synthesize_tts():
    payload = request.get_json(silent=True) or {}
    if not payload:
        payload = request.form.to_dict()

    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text to speak."}), 400

    rate = int(payload.get("rate", 180))
    _ = rate
    exaggeration = float(payload.get("exaggeration", 0.5))
    voice_id = (payload.get("voice") or "default").strip()
    if voice_id not in {"default", "neutral", "warm", "dramatic", "energetic", "calm"}:
        voice_id = "default"

    reference_file = request.files.get("reference_audio")
    reference_audio_path = None
    if reference_file and reference_file.filename:
        safe_name = secure_filename(reference_file.filename) or "reference.wav"
        suffix = Path(safe_name).suffix or ".wav"
        saved_name = f"{uuid.uuid4().hex}{suffix}"
        reference_audio_path = AUDIO_DIR / saved_name
        reference_file.save(reference_audio_path)

        trimmed_path = AUDIO_DIR / f"{uuid.uuid4().hex}.wav"
        prepared = prepare_reference_audio(reference_audio_path, trimmed_path)
        if prepared:
            reference_audio_path = trimmed_path

    filename = f"{uuid.uuid4().hex}.wav"
    output_path = AUDIO_DIR / filename

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "progress": 0}

    threading.Thread(
        target=render_job,
        args=(job_id, text, reference_audio_path, exaggeration, output_path, filename),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id, "status_url": f"/api/tts/{job_id}"}), 202


@app.route("/api/tts/<job_id>")
def get_tts_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify({"error": "Generation job not found."}), 404
        return jsonify(job)


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    preload_model()
    app.run(host="0.0.0.0", port=8000, debug=True)
