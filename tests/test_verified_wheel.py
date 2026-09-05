from __future__ import annotations

import hashlib
import http.client
import io
import json
from pathlib import Path
import runpy
import sys

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "download_verified_wheel.py"))
run = MODULE["main"]


class Response(io.BytesIO):
    status = 206

    def __init__(self, data, content_range):
        super().__init__(data)
        self.headers = {"Content-Range": content_range}


def setup_download(tmp_path, monkeypatch, *, data=b"verified bytes", prefix=b"", corrupt=False):
    name = "test-1-cp310-cp310-linux_x86_64.whl"
    lock = tmp_path / "lock.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    lock.write_text(json.dumps({"filename": name, "bytes": len(data),
                               "sha256": hashlib.sha256(data).hexdigest(),
                               "url": "https://example.invalid/wheel"}))
    if prefix:
        (cache / (name + ".prefix.part")).write_bytes(prefix)
    calls = []

    def open_range(request, *, timeout):
        assert request.full_url.startswith("https://")
        assert timeout == 20
        start, stop = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        calls.append((start, stop))
        body = data[start:stop + 1]
        if corrupt:
            body = b"x" * len(body)
        return Response(body, f"bytes {start}-{stop}/{len(data)}")

    monkeypatch.setattr(MODULE["urllib"].request, "urlopen", open_range)
    monkeypatch.setattr(MODULE["time"], "sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["download", "--lock", str(lock), "--cache-dir", str(cache), "--workers", "1"])
    return cache / name, calls, open_range


def test_resume_prefix_and_reverify_without_network(tmp_path, monkeypatch):
    final, calls, _ = setup_download(tmp_path, monkeypatch, prefix=b"verified ")
    run()
    assert final.read_bytes() == b"verified bytes"
    assert calls == [(9, 13)]
    calls.clear()
    run()
    assert calls == []


def test_incomplete_http_read_is_retried(tmp_path, monkeypatch):
    final, calls, original = setup_download(tmp_path, monkeypatch)
    attempts = 0

    def interrupted(request, *, timeout):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.IncompleteRead(b"short", 9)
        return original(request, timeout=timeout)

    monkeypatch.setattr(MODULE["urllib"].request, "urlopen", interrupted)
    run()
    assert attempts == 2 and len(calls) == 1
    assert final.read_bytes() == b"verified bytes"


def test_wrong_range_fails_after_bounded_retries(tmp_path, monkeypatch):
    final, calls, original = setup_download(tmp_path, monkeypatch)

    def wrong_range(request, *, timeout):
        response = original(request, timeout=timeout)
        response.headers["Content-Range"] = "bytes 1-14/15"
        return response

    monkeypatch.setattr(MODULE["urllib"].request, "urlopen", wrong_range)
    with pytest.raises(RuntimeError, match="bounded retries"):
        run()
    assert len(calls) == 8
    assert not final.exists()


def test_corrupt_full_wheel_is_never_accepted_or_overwritten(tmp_path, monkeypatch):
    final, calls, _ = setup_download(tmp_path, monkeypatch, corrupt=True)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        run()
    evidence = final.read_bytes()
    calls.clear()
    with pytest.raises(RuntimeError, match="preserved for inspection"):
        run()
    assert final.read_bytes() == evidence and calls == []


def test_overlapping_saved_ranges_are_rejected_before_network(tmp_path, monkeypatch):
    final, calls, _ = setup_download(tmp_path, monkeypatch, prefix=b"verified ")
    parts = final.parent / (final.name + ".parts")
    parts.mkdir()
    (parts / "0-1.part").write_bytes(b"v")
    with pytest.raises(RuntimeError, match="overlap"):
        run()
    assert calls == []


def test_saved_gap_is_completed_without_redownloading_other_ranges(tmp_path, monkeypatch):
    final, calls, _ = setup_download(tmp_path, monkeypatch, prefix=b"verified ")
    parts = final.parent / (final.name + ".parts")
    parts.mkdir()
    (parts / "11-14.part").write_bytes(b"tes")
    run()
    assert calls == [(9, 10)]
    assert final.read_bytes() == b"verified bytes"
