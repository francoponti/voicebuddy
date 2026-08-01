import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import AUDIO_DIR, app, prepare_reference_audio, split_text_for_tts


def test_homepage_renders():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Chatterbox TTS Studio" in response.data


def test_homepage_includes_reference_audio_input():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b'id="reference-audio"' in response.data


def test_voices_endpoint_lists_multiple_voice_profiles():
    client = app.test_client()
    response = client.get("/api/voices")
    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload) >= 5
    assert any(voice["id"] == "warm" for voice in payload)
    assert any(voice["id"] == "dramatic" for voice in payload)


def test_prepare_reference_audio_trims_long_wav(tmp_path):
    input_path = tmp_path / "input.wav"
    with wave.open(str(input_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 22050 * 5)

    output_path = tmp_path / "trimmed.wav"
    result = prepare_reference_audio(input_path, output_path)

    assert result == str(output_path)
    assert output_path.exists()

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnframes() <= 22050 * 3 + 1


def test_split_text_for_tts_preserves_words_across_short_chunks():
    text = "First sentence ends here. Second sentence also ends here. " * 12

    chunks = split_text_for_tts(text, max_characters=60)

    assert len(chunks) > 1
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == text.strip()


def test_tts_creates_non_empty_audio_file():
    client = app.test_client()
    response = client.post(
        "/api/tts",
        json={"text": "Hello from Chatterbox.", "voice": "Samantha", "rate": 180, "volume": 1.0, "pitch": 1.0},
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload["status_url"].startswith("/api/tts/")
    status_url = payload["status_url"]

    for _ in range(60):
        response = client.get(status_url)
        payload = response.get_json()
        if payload["status"] in {"complete", "failed"}:
            break
        time.sleep(1)

    assert payload["status"] == "complete", payload.get("error")
    assert payload["audio_url"].startswith("/audio/")
    audio_path = AUDIO_DIR / Path(payload["audio_url"].split("/", 2)[-1])
    assert audio_path.exists(), "Expected generated audio file to be created"
    assert audio_path.stat().st_size > 1000, "Expected generated audio file to be non-empty"
