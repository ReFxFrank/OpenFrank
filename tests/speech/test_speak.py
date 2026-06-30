"""Tests for the speak() voice helper (orchestration; no real audio/kokoro)."""

from __future__ import annotations

from typing import List

import pytest

from openjarvis.speech import speak as speak_mod
from openjarvis.speech.speak import get_tts_backend, speak
from openjarvis.speech.tts import TTSBackend, TTSResult


class FakeTTS(TTSBackend):
    backend_id = "fake"

    def __init__(self, audio: bytes = b"RIFFfakewav") -> None:
        self._audio = audio
        self.calls: List[dict] = []

    def synthesize(self, text, *, voice_id="DEFAULT", speed=1.0, output_format="wav"):
        self.calls.append(
            {"text": text, "voice_id": voice_id, "speed": speed, "fmt": output_format}
        )
        return TTSResult(audio=self._audio, format=output_format, voice_id=voice_id)

    def available_voices(self):
        return ["DEFAULT"]

    def health(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _no_real_audio(monkeypatch):
    # Never invoke a system audio player in tests.
    monkeypatch.setattr(speak_mod, "play_audio", lambda path: True)


def test_empty_text_returns_none():
    assert speak("   ", backend=FakeTTS()) is None


def test_no_backend_returns_none():
    assert get_tts_backend("definitely-not-a-backend") is None
    assert speak("hello", backend_name="definitely-not-a-backend") is None


def test_synthesizes_and_writes_file(tmp_path):
    out = tmp_path / "reply.wav"
    path = speak("hello there", backend=FakeTTS(), save_path=out)
    assert path == out
    assert out.read_bytes() == b"RIFFfakewav"


def test_empty_audio_returns_none(tmp_path):
    path = speak("hi", backend=FakeTTS(audio=b""), save_path=tmp_path / "x.wav")
    assert path is None


def test_blank_voice_not_passed_so_backend_default_kept(tmp_path):
    be = FakeTTS()
    speak("hi", backend=be, voice_id="", save_path=tmp_path / "a.wav")
    # voice_id omitted → backend keeps its own default ("DEFAULT").
    assert be.calls[0]["voice_id"] == "DEFAULT"


def test_explicit_voice_is_passed(tmp_path):
    be = FakeTTS()
    speak("hi", backend=be, voice_id="af_bella", save_path=tmp_path / "b.wav")
    assert be.calls[0]["voice_id"] == "af_bella"


def test_returns_path_even_when_no_player(monkeypatch, tmp_path):
    monkeypatch.setattr(speak_mod, "play_audio", lambda path: False)
    out = tmp_path / "c.wav"
    assert speak("hi", backend=FakeTTS(), save_path=out) == out


def test_synth_exception_returns_none(tmp_path):
    class Boom(FakeTTS):
        def synthesize(self, *a, **k):
            raise RuntimeError("kokoro not installed")

    assert speak("hi", backend=Boom(), save_path=tmp_path / "d.wav") is None


def test_kokoro_is_registered_local_backend():
    # The conftest wipes TTSRegistry between tests, and re-importing a cached
    # module won't re-run its @register decorator — so register explicitly
    # (as other suites do) to verify get_tts_backend builds the local kokoro
    # backend. The backend constructs without the kokoro package (load is lazy).
    from openjarvis.core.registry import TTSRegistry
    from openjarvis.speech.kokoro_tts import KokoroTTSBackend

    if not TTSRegistry.contains("kokoro"):
        TTSRegistry.register_value("kokoro", KokoroTTSBackend)
    be = get_tts_backend("kokoro")
    assert be is not None
    assert be.backend_id == "kokoro"
