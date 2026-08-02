"""
Piper TTS backend.

Runs entirely on your machine via a local neural model (ONNX) - no network
call happens during synthesis at all, only a one-time voice model download
before first use. This makes it immune to the whole category of problem
edge-tts can hit (corporate firewalls, EDR software, or any other network
policy interfering with an external WebSocket connection).

Trade-off: Piper only exposes PHONEME-level alignment, not word-level, and
reliably mapping phonemes back to the original words would mean
reverse-engineering espeak-ng's internal phonemization rules - fragile and
not worth it. Instead, we run a local Whisper pass (faster-whisper) over the
audio Piper just generated to recover word-level timestamps. This is exactly
the fallback faster-whisper was included in requirements.txt for.

SETUP (one-time, per voice):
    python -m piper.download_voices en_US-lessac-medium --download-dir models/piper
Then set production.voice_id: en_US-lessac-medium and production.tts_provider: piper
in your channel config.
"""

from __future__ import annotations

import wave
from pathlib import Path

from piper import PiperVoice

from agent.production.tts_common import LineAudio, WordTiming

PIPER_MODELS_DIR = Path("models/piper")

# "base.en" is a good speed/accuracy trade-off for short-form English
# narration. faster-whisper downloads this once (~140MB) from Hugging Face
# and caches it locally - a one-time download, not needed per line.
WHISPER_MODEL_SIZE = "base.en"

_whisper_model = None  # lazy singleton - loading it fresh per line would be slow
_piper_voice_cache: dict[str, PiperVoice] = {}  # avoid reloading the ONNX model per line


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def _resolve_model_path(voice: str) -> Path:
    model_path = PIPER_MODELS_DIR / f"{voice}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Piper voice model not found at {model_path}.\n"
            f"Download it first:\n"
            f"  python -m piper.download_voices {voice} --download-dir {PIPER_MODELS_DIR}"
        )
    return model_path


def _get_piper_voice(voice: str) -> PiperVoice:
    if voice not in _piper_voice_cache:
        model_path = _resolve_model_path(voice)
        _piper_voice_cache[voice] = PiperVoice.load(str(model_path))
    return _piper_voice_cache[voice]


def _transcribe_word_timings(audio_path: Path) -> list[WordTiming]:
    model = _get_whisper_model()
    segments, _info = model.transcribe(str(audio_path), word_timestamps=True)

    word_timings: list[WordTiming] = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            word_timings.append(
                WordTiming(
                    word=word.word.strip(),
                    start_seconds=round(word.start, 3),
                    end_seconds=round(word.end, 3),
                )
            )
    return word_timings


async def synthesize_line(text: str, voice: str, output_path: Path) -> LineAudio:
    """Synthesize one line with Piper (local, synchronous under the hood),
    then run local Whisper transcription on the result to recover word-level
    timestamps. Declared async so voiceover.py can call this backend and the
    edge-tts backend identically."""
    piper_voice = _get_piper_voice(voice)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        piper_voice.synthesize_wav(text, wav_file)

    with wave.open(str(output_path), "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()

    word_timings = _transcribe_word_timings(output_path)

    return LineAudio(
        line_number=-1,  # caller fills these in
        label="",
        text=text,
        audio_path=output_path,
        duration_seconds=round(duration, 2),
        word_timings=word_timings,
    )
