"""Download the official MSI using bounded 1 MiB ranges and verified coverage.

Preserve older partial downloads, fill only uncovered spans, then validate the
official SHA-256. This handles slow/resetting long GitHub release connections.
No TLS verification is disabled and no third-party mirror is used.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import re
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1] / "cache" / "installers"
URL = "https://github.com/microsoft/WSL/releases/download/2.7.13/wsl.2.7.13.0.x64.msi"
SIZE = 258985984
SHA = "a3505a50f4cc585551d11d9de824ba4375448d7a68f2e71d3fb315fa986fc754"
LOCAL = threading.local()


def existing_spans():
    spans = []
    prefix = ROOT / "wsl.2.7.13.prefix.part"
    if prefix.exists():
        spans.append((0, prefix.stat().st_size, prefix))
    for path in ROOT.glob("wsl.2.7.13.range-*-*.part"):
        start, end = map(int, re.search(r"range-(\d+)-(\d+)\.part", path.name).groups())
        length = path.stat().st_size
        if length > end - start + 1:
            raise RuntimeError(f"Oversize old range: {path}")
        if length:
            spans.append((start, start + length, path))
        remainder = path.with_name(path.name + ".remainder")
        if remainder.exists() and remainder.stat().st_size:
            stop = start + length + remainder.stat().st_size
            if stop > end + 1:
                raise RuntimeError("Old remainder overlaps next range")
            spans.append((start + length, stop, remainder))
    tail = ROOT / "wsl.2.7.13.dotnet-tail.part"
    if tail.exists() and tail.stat().st_size == SIZE - 251297792:
        spans.append((251297792, SIZE, tail))
    for path in (ROOT / "wsl-chunks").glob("*.part"):
        start, end = map(int, path.stem.split("-"))
        if path.stat().st_size == end - start:
            spans.append((start, end, path))
    return sorted(spans)


def download(span):
    start, end = span
    destination = ROOT / "wsl-chunks" / f"{start}-{end}.part"
    session = getattr(LOCAL, "session", None)
    if session is None:
        session = LOCAL.session = requests.Session()
    error = None
    for attempt in range(8):
        try:
            response = session.get(URL, headers={"Range": f"bytes={start}-{end-1}"}, timeout=(15, 20))
            response.raise_for_status()
            expected = f"bytes {start}-{end-1}/{SIZE}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise RuntimeError(f"Wrong range response: {response.status_code}, {response.headers.get('Content-Range')}")
            if len(response.content) != end - start:
                raise RuntimeError("Truncated range")
            destination.write_bytes(response.content)
            return end - start
        except (requests.RequestException, RuntimeError) as exc:
            error = exc
            time.sleep(min(5, attempt + 1))
    raise RuntimeError(f"Range {start}-{end} failed: {error}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "wsl-chunks").mkdir(exist_ok=True)
    final = ROOT / "wsl.2.7.13.0.x64.msi"
    if final.exists():
        with final.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest == SHA:
            print("Official MSI already verified.", flush=True)
            return
        raise RuntimeError("Existing MSI has a different hash; inspect before proceeding")
    spans = existing_spans()
    gaps, cursor = [], 0
    for start, end, _ in spans:
        if start < cursor:
            raise RuntimeError("Overlapping saved spans; inspect before assembling")
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < SIZE:
        gaps.append((cursor, SIZE))
    chunks = [(start, min(start + 1024 * 1024, end)) for a, end in gaps for start in range(a, end, 1024 * 1024)]
    print(f"Missing {sum(b-a for a,b in chunks)} bytes in {len(chunks)} bounded ranges", flush=True)
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download, chunk) for chunk in chunks]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            total += future.result()
            print(f"range {index}/{len(chunks)}; downloaded {total/1024/1024:.1f} MiB", flush=True)
    cursor = 0
    with final.open("xb") as output:
        for start, end, path in existing_spans():
            if start != cursor:
                raise RuntimeError(f"Coverage error at {cursor}: next span starts {start}")
            with path.open("rb") as source:
                while data := source.read(1024 * 1024):
                    output.write(data)
            cursor = end
    if cursor != SIZE:
        raise RuntimeError("Assembled length mismatch")
    with final.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != SHA:
        raise RuntimeError("Official MSI SHA-256 mismatch; preserve artifacts for diagnosis")
    print(f"VERIFIED {digest} {final}", flush=True)


if __name__ == "__main__":
    main()
