"""
Tests for the orchestration layer (agent/production/voiceover.py) - dispatch
to the correct backend, playback ordering, and timeline math. These use a
completely fake backend (not edge-tts or Piper) so they test ONLY the
orchestration logic, independent of any specific TTS implementation. Backend
specifics are covered in test_tts_edge.py and test_tts_piper.py.

Run with:
    pytest tests/test_voiceover.py -v
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.config import TTSProvider
from agent.production import voiceover as vo
from agent.production.tts_common import LineAudio, WordTiming


async def _fake_synth(text: str, voice: str, output_path: Path) -> LineAudio:
    """A fake backend: 1 second per line, one fake word, no real files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"fake-audio")
    return LineAudio(
        line_number=-1,
        label="",
        text=text,
        audio_path=output_path,
        duration_seconds=1.0,
        word_timings=[WordTiming(word=text.split()[0], start_seconds=0.0, end_seconds=0.5)],
    )


def test_get_backend_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="No TTS backend"):
        vo._get_backend("not_a_real_provider")


def test_synthesize_script_uses_correct_backend_and_extension(tmp_path):
    fake_backends = {TTSProvider.EDGE_TTS: (_fake_synth, "mp3")}
    with patch.object(vo, "_BACKENDS", fake_backends):
        clips = asyncio.run(
            vo.synthesize_script(
                hook="Hook line",
                scenes=[(1, "Scene one"), (2, "Scene two")],
                cta="Call to action",
                voice="fake-voice",
                output_dir=tmp_path,
                provider=TTSProvider.EDGE_TTS,
            )
        )

    assert [c.label for c in clips] == ["hook", "scene_1", "scene_2", "cta"]
    assert [c.line_number for c in clips] == [0, 1, 2, 999]
    assert all(c.audio_path.suffix == ".mp3" for c in clips)
    assert (tmp_path / "00_hook.mp3").exists()


def test_synthesize_script_uses_wav_extension_for_piper(tmp_path):
    fake_backends = {TTSProvider.PIPER: (_fake_synth, "wav")}
    with patch.object(vo, "_BACKENDS", fake_backends):
        clips = asyncio.run(
            vo.synthesize_script(
                hook="Hook line",
                scenes=[(1, "Scene one")],
                cta="Call to action",
                voice="en_US-lessac-medium",
                output_dir=tmp_path,
                provider=TTSProvider.PIPER,
            )
        )

    assert all(c.audio_path.suffix == ".wav" for c in clips)


def test_total_duration_and_timeline_math(tmp_path):
    fake_backends = {TTSProvider.EDGE_TTS: (_fake_synth, "mp3")}
    with patch.object(vo, "_BACKENDS", fake_backends):
        clips = asyncio.run(
            vo.synthesize_script(
                hook="Hook line",
                scenes=[(1, "Scene one"), (2, "Scene two")],
                cta="Call to action",
                voice="fake-voice",
                output_dir=tmp_path,
                provider=TTSProvider.EDGE_TTS,
            )
        )

    # 4 clips * 1.0s each from the fake backend
    assert vo.total_duration(clips) == 4.0

    timeline = vo.build_timeline(clips)
    assert timeline[0]["start"] == 0.0
    assert timeline[0]["end"] == 1.0
    assert timeline[1]["start"] == 1.0
    assert timeline[2]["start"] == 2.0
    assert timeline[3]["start"] == 3.0
    assert timeline[3]["label"] == "cta"

    # word timings offset to the FULL track, not clip-relative
    assert timeline[1]["word_timings"][0]["start"] == 1.0
    assert timeline[2]["word_timings"][0]["start"] == 2.0
