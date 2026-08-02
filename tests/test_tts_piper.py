"""
Tests for the Piper backend (agent/production/tts_piper.py).

PiperVoice and the Whisper model are mocked (downloading real models needs
network access this sandbox doesn't have to Hugging Face - on your machine,
these download for real, once, and are then cached). Real wave-file I/O is
used, so duration-reading is tested against genuine audio, not a faked
number - same principle as the edge-tts tests.

Run with:
    pytest tests/test_tts_piper.py -v
"""

import asyncio
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.production import tts_piper


class _FakePiperVoice:
    """Stands in for piper.PiperVoice - writes real silent frames into
    whatever wave.Wave_write object our code opens, so duration-reading
    downstream is tested against a real audio file."""

    def synthesize_wav(self, text, wav_file, **kwargs):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        num_frames = int(22050 * 1.5)  # 1.5 seconds of silence
        wav_file.writeframes(b"\x00\x00" * num_frames)


class _FakePiperVoiceClass:
    @staticmethod
    def load(*args, **kwargs):
        return _FakePiperVoice()


class _FakeWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class _FakeSegment:
    def __init__(self, words):
        self.words = words


class _FakeWhisperModel:
    def __init__(self, fake_words):
        self._fake_words = fake_words

    def transcribe(self, audio_path, word_timestamps=True):
        segment = _FakeSegment(words=self._fake_words)
        info = object()
        return [segment], info


@pytest.fixture(autouse=True)
def _reset_piper_caches():
    """The real module caches loaded voices/models in module-level dicts -
    reset them between tests so one test's mock doesn't leak into another."""
    tts_piper._piper_voice_cache.clear()
    tts_piper._whisper_model = None
    yield
    tts_piper._piper_voice_cache.clear()
    tts_piper._whisper_model = None


def test_resolve_model_path_raises_clear_error_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tts_piper, "PIPER_MODELS_DIR", tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="download_voices"):
        tts_piper._resolve_model_path("en_US-lessac-medium")


def test_synthesize_line_writes_real_audio_and_gets_duration(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-lessac-medium.onnx").write_bytes(b"fake-model-bytes")
    monkeypatch.setattr(tts_piper, "PIPER_MODELS_DIR", models_dir)
    monkeypatch.setattr(tts_piper, "PiperVoice", _FakePiperVoiceClass)

    fake_words = [
        _FakeWord("Hello", 0.0, 0.5),
        _FakeWord("world", 0.5, 1.0),
    ]
    with patch.object(tts_piper, "_get_whisper_model", return_value=_FakeWhisperModel(fake_words)):
        output_path = tmp_path / "out" / "test.wav"
        result = asyncio.run(
            tts_piper.synthesize_line("Hello world", "en_US-lessac-medium", output_path)
        )

    assert output_path.exists()
    # Real duration read from the actual wav file we wrote (1.5s of silence)
    assert result.duration_seconds == pytest.approx(1.5, abs=0.01)
    assert len(result.word_timings) == 2
    assert result.word_timings[0].word == "Hello"
    assert result.word_timings[0].start_seconds == 0.0
    assert result.word_timings[1].word == "world"
    assert result.word_timings[1].end_seconds == 1.0


def test_synthesize_line_caches_loaded_voice(tmp_path, monkeypatch):
    """Confirms we don't reload the ONNX model from disk for every line -
    that would be slow across a whole script's worth of lines."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-lessac-medium.onnx").write_bytes(b"fake-model-bytes")
    monkeypatch.setattr(tts_piper, "PIPER_MODELS_DIR", models_dir)

    load_call_count = {"n": 0}

    class CountingFakePiperVoiceClass:
        @staticmethod
        def load(*args, **kwargs):
            load_call_count["n"] += 1
            return _FakePiperVoice()

    monkeypatch.setattr(tts_piper, "PiperVoice", CountingFakePiperVoiceClass)

    with patch.object(tts_piper, "_get_whisper_model", return_value=_FakeWhisperModel([])):
        asyncio.run(tts_piper.synthesize_line("First line", "en_US-lessac-medium", tmp_path / "a.wav"))
        asyncio.run(tts_piper.synthesize_line("Second line", "en_US-lessac-medium", tmp_path / "b.wav"))

    assert load_call_count["n"] == 1  # loaded once, reused for the second line


def test_transcribe_word_timings_skips_segments_with_no_words(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "en_US-lessac-medium.onnx").write_bytes(b"fake-model-bytes")
    monkeypatch.setattr(tts_piper, "PIPER_MODELS_DIR", models_dir)
    monkeypatch.setattr(tts_piper, "PiperVoice", _FakePiperVoiceClass)

    class EmptyWordsWhisperModel:
        def transcribe(self, audio_path, word_timestamps=True):
            segment = _FakeSegment(words=None)  # e.g. silence/no-speech segment
            return [segment], object()

    with patch.object(tts_piper, "_get_whisper_model", return_value=EmptyWordsWhisperModel()):
        output_path = tmp_path / "out" / "test.wav"
        result = asyncio.run(
            tts_piper.synthesize_line("...", "en_US-lessac-medium", output_path)
        )

    assert result.word_timings == []
