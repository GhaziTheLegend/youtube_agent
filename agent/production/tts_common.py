"""
Shared data types for TTS backends (tts_edge.py, tts_piper.py). Both backends
return LineAudio objects with this exact shape, which is what lets
voiceover.py orchestrate either one identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WordTiming:
    word: str
    start_seconds: float
    end_seconds: float


@dataclass
class LineAudio:
    line_number: int   # 0 = hook, 1..N = scenes, 999 = cta
    label: str          # "hook" / "scene_1" / "cta" - used for filenames + debugging
    text: str
    audio_path: Path
    duration_seconds: float
    word_timings: list[WordTiming] = field(default_factory=list)
