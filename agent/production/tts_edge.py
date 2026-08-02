"""
edge-tts backend.

Free, sounds natural, and gives word-level timing directly from its own
boundary events - no separate transcription pass needed. The catch: it's an
unofficial API (a wrapper around Microsoft Edge's "Read Aloud" feature), so
some networks - corporate firewalls/EDR software in particular - can
interfere with its WebSocket handshake in ways outside your control. If
that happens to you, see tts_piper.py for a fully local, network-free
alternative.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import edge_tts
from mutagen.mp3 import MP3

from agent.production.tts_common import LineAudio, WordTiming

if sys.platform == "win32":
    # Windows' default asyncio event loop ("Proactor") has a long-standing,
    # documented history of issues with aiohttp's WebSocket+SSL handshake
    # specifically (WinError 64 "network name is no longer available"),
    # independent of whether the remote server is reachable. The "Selector"
    # event loop doesn't have this issue. Must be set before any event loop
    # is created, hence this runs at import time.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Even past the fix above, external services can have a genuine transient
# hiccup. Retrying (rather than crashing the whole batch on one bad request)
# is standard practice for any pipeline step that talks to an external
# service.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


async def synthesize_line(text: str, voice: str, output_path: Path) -> LineAudio:
    """Synthesize one line of text to an mp3 file via edge-tts, capturing
    word timings as they're generated. Retries on transient connection
    failures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await _synthesize_line_once(text, voice, output_path)
        except (OSError, ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_SECONDS * attempt
                print(
                    f"⚠️  Voiceover request failed ({e.__class__.__name__}: {e}). "
                    f"Retrying in {wait}s... (attempt {attempt}/{MAX_RETRIES})"
                )
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"Failed to synthesize line after {MAX_RETRIES} attempts: {text[:60]!r}. "
        f"Last error: {last_error}\n"
        f"If this keeps happening (especially with a WebSocket-specific error "
        f"like a connection reset right after the handshake starts), it's "
        f"often a corporate/school network security policy interfering with "
        f"WebSocket upgrades to less-common external hosts - consider "
        f"switching production.tts_provider to 'piper' in your channel "
        f"config, which runs fully offline with no network calls at all."
    ) from last_error


async def _synthesize_line_once(text: str, voice: str, output_path: Path) -> LineAudio:
    communicate = edge_tts.Communicate(text, voice)
    word_timings: list[WordTiming] = []

    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # edge-tts reports offset/duration in 100-nanosecond units
                start = chunk["offset"] / 1e7
                duration = chunk["duration"] / 1e7
                word_timings.append(
                    WordTiming(
                        word=chunk["text"],
                        start_seconds=round(start, 3),
                        end_seconds=round(start + duration, 3),
                    )
                )

    # Measure the ACTUAL rendered duration from the written file - never
    # trust the sum of word-boundary durations, which can undercount
    # leading/trailing silence edge-tts adds.
    duration = MP3(output_path).info.length

    return LineAudio(
        line_number=-1,  # caller fills these in
        label="",
        text=text,
        audio_path=output_path,
        duration_seconds=round(duration, 2),
        word_timings=word_timings,
    )
