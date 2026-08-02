"""
Tests for the integrity-verified Piper voice downloader.

These simulate exactly what happened on the real network this was built
for: a download that "completes" (no exception from the HTTP layer) but
delivers fewer bytes than the server declared via Content-Length - and
confirm we catch that, retry, and only accept a file once it's verified
complete.

Run with:
    pytest tests/test_tts_downloader.py -v
"""

from unittest.mock import patch

import pytest
import requests

from agent.production import tts_downloader as dl


class _FakeResponse:
    """Stands in for requests.Response when used as `with requests.get(...) as r:`."""

    def __init__(self, content: bytes, declared_length: int | None = None, ok: bool = True):
        self._content = content
        self._ok = ok
        self.headers = {}
        if declared_length is not None:
            self.headers["content-length"] = str(declared_length)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if not self._ok:
            raise requests.exceptions.HTTPError("simulated bad status")

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]


def test_download_succeeds_when_bytes_match_content_length(tmp_path):
    full_content = b"x" * 1000
    fake = _FakeResponse(full_content, declared_length=1000)

    with patch.object(dl.requests, "get", return_value=fake):
        dl._download_one_file("http://fake/model.onnx", tmp_path / "model.onnx", max_retries=3)

    dest = tmp_path / "model.onnx"
    assert dest.exists()
    assert dest.read_bytes() == full_content
    assert not (tmp_path / "model.onnx.partial").exists()  # temp file cleaned up


def test_download_retries_and_recovers_from_truncated_transfer(tmp_path, monkeypatch):
    """Reproduces the real bug: first attempt is truncated (server said 1000
    bytes, only 400 arrived) - this must be detected and retried, not
    silently accepted like piper's own downloader did."""
    truncated = _FakeResponse(b"x" * 400, declared_length=1000)   # bad: mismatch
    complete = _FakeResponse(b"y" * 1000, declared_length=1000)   # good: matches

    call_count = {"n": 0}

    def fake_get(*args, **kwargs):
        call_count["n"] += 1
        return truncated if call_count["n"] == 1 else complete

    monkeypatch.setattr(dl.time, "sleep", lambda _seconds: None)  # skip real backoff wait

    with patch.object(dl.requests, "get", side_effect=fake_get):
        dl._download_one_file("http://fake/model.onnx", tmp_path / "model.onnx", max_retries=3)

    assert call_count["n"] == 2  # failed once (truncated), succeeded on retry
    dest = tmp_path / "model.onnx"
    assert dest.read_bytes() == b"y" * 1000  # the COMPLETE version, not the truncated one


def test_download_raises_clear_error_after_max_retries_all_truncated(tmp_path, monkeypatch):
    always_truncated = _FakeResponse(b"x" * 400, declared_length=1000)
    monkeypatch.setattr(dl.time, "sleep", lambda _seconds: None)

    with patch.object(dl.requests, "get", return_value=always_truncated):
        with pytest.raises(RuntimeError, match="truncated"):
            dl._download_one_file("http://fake/model.onnx", tmp_path / "model.onnx", max_retries=3)

    # no bad file left behind where a good one would be expected
    assert not (tmp_path / "model.onnx").exists()


def test_download_never_leaves_partial_file_after_failure(tmp_path, monkeypatch):
    always_truncated = _FakeResponse(b"x" * 400, declared_length=1000)
    monkeypatch.setattr(dl.time, "sleep", lambda _seconds: None)

    with patch.object(dl.requests, "get", return_value=always_truncated):
        with pytest.raises(RuntimeError):
            dl._download_one_file("http://fake/model.onnx", tmp_path / "model.onnx", max_retries=2)

    assert not (tmp_path / "model.onnx.partial").exists()


def test_download_voice_rejects_malformed_voice_name(tmp_path):
    with pytest.raises(ValueError, match="doesn't match the expected pattern"):
        dl.download_voice("not-a-valid-voice-name-format!!", tmp_path)


def test_download_voice_downloads_both_model_and_config(tmp_path, monkeypatch):
    """Confirms both the .onnx and .onnx.json get requested with correct
    voice-code-based filenames."""
    requested_urls = []

    def fake_download_one_file(url, dest_path, max_retries=5):
        requested_urls.append(url)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(b"fake")

    monkeypatch.setattr(dl, "_download_one_file", fake_download_one_file)

    dl.download_voice("en_US-lessac-medium", tmp_path)

    assert (tmp_path / "en_US-lessac-medium.onnx").exists()
    assert (tmp_path / "en_US-lessac-medium.onnx.json").exists()
    assert any(url.endswith(".onnx?download=true") for url in requested_urls)
    assert any(url.endswith(".onnx.json?download=true") for url in requested_urls)
