"""Speak text aloud via a local TTS backend — the assistant's voice.

Thin, best-effort wrapper used by ``jarvis chat --speak``: synthesize a reply
with a TTS backend (defaulting to the **local** Kokoro voice so it works offline
under ``local_only``), write it to a wav, and play it through whatever audio
player is on PATH. Everything is guarded — if TTS or audio isn't available it
returns ``None`` and the caller carries on silently (a missing voice must never
break the chat).

Audio note: on WSL2, playback needs WSLg (Windows 11) or a PulseAudio bridge;
without one, the wav is still written and its path is returned/logged.
"""

from __future__ import annotations

import importlib
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from openjarvis.core.registry import TTSRegistry
from openjarvis.speech.tts import TTSBackend

logger = logging.getLogger(__name__)

# Backend name -> module that registers it via @TTSRegistry.register(...).
# kokoro is local; cartesia/openai_tts are cloud (blocked under local_only).
_BACKEND_MODULES = {
    "kokoro": "openjarvis.speech.kokoro_tts",
    "cartesia": "openjarvis.speech.cartesia_tts",
    "openai_tts": "openjarvis.speech.openai_tts",
}

# Audio players tried in order (paplay = PulseAudio, common under WSLg).
_PLAYERS = ("paplay", "aplay", "ffplay", "play", "afplay", "cvlc")


def get_tts_backend(name: str = "kokoro") -> Optional[TTSBackend]:
    """Instantiate a registered TTS backend by name, or None if unavailable."""
    key = (name or "kokoro").strip().lower()
    module = _BACKEND_MODULES.get(key)
    if module:
        try:
            importlib.import_module(module)  # triggers TTSRegistry.register
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not import TTS module %s: %s", module, exc)
    if not TTSRegistry.contains(key):
        return None
    try:
        return TTSRegistry.create(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("TTS backend %r could not be created: %s", key, exc)
        return None


def play_audio(path: Path) -> bool:
    """Play an audio file with the first available system player. Best-effort."""
    for player in _PLAYERS:
        exe = shutil.which(player)
        if not exe:
            continue
        if player == "ffplay":
            args = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
        elif player == "cvlc":
            args = [exe, "--play-and-exit", "--intf", "dummy", str(path)]
        else:
            args = [exe, str(path)]
        try:
            subprocess.run(
                args,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("player %s failed: %s", player, exc)
            continue
    return False


def speak(
    text: str,
    *,
    backend: Optional[TTSBackend] = None,
    backend_name: str = "kokoro",
    voice_id: str = "",
    speed: float = 1.0,
    save_path: Optional[Path] = None,
    play: bool = True,
) -> Optional[Path]:
    """Synthesize *text* and (optionally) play it. Returns the wav path or None.

    Never raises — any failure (no backend, missing kokoro package, no audio
    device) logs at debug and returns None so the caller continues silently.
    """
    text = (text or "").strip()
    if not text:
        return None

    be = backend or get_tts_backend(backend_name)
    if be is None:
        logger.debug("no TTS backend available (%s)", backend_name)
        return None

    # Only pass voice_id when set, so each backend keeps its own default
    # (e.g. Kokoro's "af_heart") rather than being overridden with "".
    kwargs = {"speed": speed, "output_format": "wav"}
    if voice_id:
        kwargs["voice_id"] = voice_id
    try:
        result = be.synthesize(text, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.debug("TTS synthesis failed: %s", exc)
        return None
    if not result.audio:
        return None

    if save_path is None:
        save_path = (
            Path(tempfile.gettempdir()) / f"openjarvis_tts.{result.format or 'wav'}"
        )
    save_path = Path(save_path)
    try:
        result.save(save_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not write audio to %s: %s", save_path, exc)
        return None

    if play and not play_audio(save_path):
        logger.info("Spoken reply written to %s (no audio player found)", save_path)
    return save_path


__all__ = ["get_tts_backend", "play_audio", "speak"]
