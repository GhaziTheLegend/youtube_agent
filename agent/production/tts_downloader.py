"""
Integrity-verified downloader for Piper voice models.

piper.download_voices' own downloader (urlopen + shutil.copyfileobj) never
checks the number of bytes actually received against the server's declared
Content-Length. On a network that silently truncates long-lived connections
partway through (which this project already hit with edge-tts's WebSocket
handshake), that means a truncated download can complete "successfully"
from Python's perspective while producing a corrupt, unusable file - which
is exactly what happened here: two separate attempts produced two
different, both-wrong file sizes for the same ~60MB model.

This downloads to a temporary file, verifies the byte count matches
Content-Length, retries with backoff on mismatch, and only moves the file
into its final place once confirmed complete (atomic rename - so a failed
attempt never leaves a bad file where the "real" one is expected to be).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests

VOICE_PATTERN = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)
URL_FORMAT = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "{lang_family}/{lang_code}/{voice_name}/{voice_quality}/"
    "{lang_code}-{voice_name}-{voice_quality}{extension}?download=true"
)


class IncompleteDownloadError(RuntimeError):
    pass


def _download_one_file(url: str, dest_path: Path, max_retries: int = 5) -> None:
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".partial")
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                expected_size = int(content_length) if content_length is not None else None

                bytes_written = 0
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):  # 1MB
                        f.write(chunk)
                        bytes_written += len(chunk)

            if expected_size is not None and bytes_written != expected_size:
                raise IncompleteDownloadError(
                    f"Downloaded {bytes_written:,} bytes but the server declared "
                    f"{expected_size:,} bytes - the connection was truncated."
                )

            tmp_path.replace(dest_path)  # only "commit" once verified complete
            print(f"✅ {dest_path.name}: {bytes_written:,} bytes, verified complete")
            return

        except (requests.RequestException, IncompleteDownloadError, OSError) as e:
            last_error = e
            tmp_path.unlink(missing_ok=True)
            if attempt < max_retries:
                wait = 2 * attempt
                print(f"⚠️  Attempt {attempt}/{max_retries} failed ({e}). Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(
        f"Failed to download {dest_path.name} after {max_retries} attempts, all "
        f"truncated/failed. Last error: {last_error}\n"
        f"Your network appears to be cutting off long downloads inconsistently. "
        f"As a manual fallback, download this URL directly in a browser (which "
        f"shows a real progress bar and won't silently truncate):\n  {url}\n"
        f"Then save it as:\n  {dest_path}"
    )


def download_voice(voice: str, download_dir: Path, max_retries: int = 5) -> None:
    """Download a Piper voice's .onnx model and .onnx.json config, verifying
    each file's byte count against the server's Content-Length before
    trusting it. Raises if either file can't be fully retrieved."""
    match = VOICE_PATTERN.match(voice.strip())
    if not match:
        raise ValueError(
            f"Voice {voice!r} doesn't match the expected pattern "
            f"<lang>_<REGION>-<name>-<quality>, e.g. 'en_US-lessac-medium'"
        )

    lang_family = match.group("lang_family")
    lang_code = f"{lang_family}_{match.group('lang_region')}"
    voice_name = match.group("voice_name")
    voice_quality = match.group("voice_quality")
    format_args = dict(
        lang_family=lang_family, lang_code=lang_code,
        voice_name=voice_name, voice_quality=voice_quality,
    )
    voice_code = f"{lang_code}-{voice_name}-{voice_quality}"

    download_dir = Path(download_dir)
    for extension in (".onnx", ".onnx.json"):
        url = URL_FORMAT.format(extension=extension, **format_args)
        dest = download_dir / f"{voice_code}{extension}"
        _download_one_file(url, dest, max_retries=max_retries)
