"""
Tests for the edge-tts backend (agent/production/tts_edge.py).

The network call is mocked, but real ffmpeg-generated audio files are used,
so duration-reading (mutagen) is tested against genuine audio, not a faked
number. Only the network boundary is faked.

Run with:
    pytest tests/test_tts_edge.py -v
"""

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.production import tts_edge


def _make_silent_mp3(path: Path, seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", str(seconds), "-q:a", "9", str(path),
        ],
        check=True, capture_output=True,
    )


def _fake_communicate_class(audio_bytes: bytes, word_events: list[dict]):
    class FakeCommunicate:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            half = len(audio_bytes) // 2
            yield {"type": "audio", "data": audio_bytes[:half]}
            yield {"type": "audio", "data": audio_bytes[half:]}
            for event in word_events:
                yield event

    return FakeCommunicate


def test_synthesize_line_reads_real_duration_and_word_timings(tmp_path):
    real_mp3 = tmp_path / "source.mp3"
    _make_silent_mp3(real_mp3, 2.0)
    audio_bytes = real_mp3.read_bytes()

    word_events = [
        {"type": "WordBoundary", "offset": 0, "duration": 5_000_000, "text": "Hello"},
        {"type": "WordBoundary", "offset": 5_000_000, "duration": 5_000_000, "text": "world"},
    ]
    fake_cls = _fake_communicate_class(audio_bytes, word_events)

    output_path = tmp_path / "out" / "test.mp3"
    with patch("agent.production.tts_edge.edge_tts.Communicate", fake_cls):
        result = asyncio.run(tts_edge.synthesize_line("Hello world", "en-US-GuyNeural", output_path))

    assert output_path.exists()
    assert result.duration_seconds == pytest.approx(2.0, abs=0.15)
    assert len(result.word_timings) == 2
    assert result.word_timings[0] == tts_edge.WordTiming("Hello", 0.0, 0.5)
    assert result.word_timings[1] == tts_edge.WordTiming("world", 0.5, 1.0)


def test_synthesize_line_retries_on_transient_connection_error(tmp_path, monkeypatch):
    real_mp3 = tmp_path / "source.mp3"
    _make_silent_mp3(real_mp3, 1.0)
    audio_bytes = real_mp3.read_bytes()

    call_count = {"n": 0}

    class FlakyThenWorksCommunicate:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("[WinError 64] The specified network name is no longer available")
            yield {"type": "audio", "data": audio_bytes}
            yield {"type": "WordBoundary", "offset": 0, "duration": 10_000_000, "text": "word"}

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(tts_edge.asyncio, "sleep", _no_sleep)

    output_path = tmp_path / "out" / "retry_test.mp3"
    with patch("agent.production.tts_edge.edge_tts.Communicate", FlakyThenWorksCommunicate):
        result = asyncio.run(tts_edge.synthesize_line("Hello there", "en-US-GuyNeural", output_path))

    assert call_count["n"] == 2
    assert output_path.exists()
    assert result.duration_seconds == pytest.approx(1.0, abs=0.15)


def test_synthesize_line_gives_up_after_max_retries(tmp_path, monkeypatch):
    class AlwaysFailsCommunicate:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            raise OSError("[WinError 10054] An existing connection was forcibly closed by the remote host")
            yield  # pragma: no cover - unreachable, makes this an async generator

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(tts_edge.asyncio, "sleep", _no_sleep)

    output_path = tmp_path / "out" / "always_fails.mp3"
    with patch("agent.production.tts_edge.edge_tts.Communicate", AlwaysFailsCommunicate):
        with pytest.raises(RuntimeError, match="Failed to synthesize line"):
            asyncio.run(tts_edge.synthesize_line("Hello there", "en-US-GuyNeural", output_path))
