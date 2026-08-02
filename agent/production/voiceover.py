"""
Voiceover module - Phase 3A.

Public orchestration layer: turns a script into per-line audio clips plus a
timeline, regardless of which TTS backend actually generates the audio. The
backend is selected via config.production.tts_provider - see tts_edge.py
(free, high quality, network-dependent) and tts_piper.py (free, fully
local/offline) for the actual synthesis implementations.

Synthesizing per-line (not one big audio blob) is what makes the rest of
production possible: because we know each clip's exact duration, the
timeline (when each scene's visual should appear) is just the running sum
of clip durations - no manual alignment needed.
"""

from __future__ import annotations

from pathlib import Path

from agent.config import TTSProvider
from agent.production.tts_common import LineAudio, WordTiming  # re-exported for convenience

_BACKENDS = None  # lazy-populated so importing voiceover.py doesn't require
                   # BOTH edge_tts and piper to be installed if you only use one


def _get_backend(provider: TTSProvider):
    global _BACKENDS
    if _BACKENDS is None:
        from agent.production import tts_edge, tts_piper

        _BACKENDS = {
            TTSProvider.EDGE_TTS: (tts_edge.synthesize_line, "mp3"),
            TTSProvider.PIPER: (tts_piper.synthesize_line, "wav"),
        }
    if provider not in _BACKENDS:
        raise ValueError(
            f"No TTS backend implemented for provider={provider!r}. "
            f"Available: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[provider]


async def synthesize_script(
    hook: str,
    scenes: list[tuple[int, str]],  # (line_number, text) pairs, in order
    cta: str,
    voice: str,
    output_dir: Path | str,
    provider: TTSProvider = TTSProvider.EDGE_TTS,
) -> list[LineAudio]:
    """Synthesize every line of a script as separate audio clips, in
    playback order: [hook, scene_1, ..., scene_N, cta]."""
    synth, ext = _get_backend(provider)
    output_dir = Path(output_dir)
    results: list[LineAudio] = []

    hook_audio = await synth(hook, voice, output_dir / f"00_hook.{ext}")
    hook_audio.line_number = 0
    hook_audio.label = "hook"
    results.append(hook_audio)

    for line_number, text in scenes:
        path = output_dir / f"{line_number:02d}_scene.{ext}"
        scene_audio = await synth(text, voice, path)
        scene_audio.line_number = line_number
        scene_audio.label = f"scene_{line_number}"
        results.append(scene_audio)

    cta_audio = await synth(cta, voice, output_dir / f"99_cta.{ext}")
    cta_audio.line_number = 999
    cta_audio.label = "cta"
    results.append(cta_audio)

    return results


def total_duration(clips: list[LineAudio]) -> float:
    return round(sum(c.duration_seconds for c in clips), 2)


def build_timeline(clips: list[LineAudio]) -> list[dict]:
    """Compute the start/end time of each clip within the final concatenated
    audio track. visuals.py (3B) and assembly.py (3C) use this to know
    exactly when each scene's visual and captions should appear."""
    timeline = []
    cursor = 0.0
    for clip in clips:
        timeline.append(
            {
                "label": clip.label,
                "line_number": clip.line_number,
                "text": clip.text,
                "start": round(cursor, 2),
                "end": round(cursor + clip.duration_seconds, 2),
                "audio_path": str(clip.audio_path),
                "word_timings": [
                    {
                        "word": w.word,
                        "start": round(cursor + w.start_seconds, 3),
                        "end": round(cursor + w.end_seconds, 3),
                    }
                    for w in clip.word_timings
                ],
            }
        )
        cursor += clip.duration_seconds
    return timeline
