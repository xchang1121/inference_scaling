"""Resume a hash-locked HTTPS wheel using bounded ranges; Python 3.10 stdlib only."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import re
import time
import urllib.request
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    name, size, sha, url = (lock[k] for k in ("filename", "bytes", "sha256", "url"))
    if (Path(name).name != name or not name.endswith(".whl") or not url.startswith("https://")
            or not re.fullmatch(r"[a-f0-9]{64}", sha) or size < 1 or not 1 <= args.workers <= 16):
        raise ValueError("invalid wheel lock or worker count")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    final = args.cache_dir / name
    if final.exists():
        if final.stat().st_size != size or digest_file(final) != sha:
            raise RuntimeError("Existing wheel does not match lock; preserved for inspection")
        print(f"VERIFIED existing {name} {sha}", flush=True)
        return
    parts = args.cache_dir / (name + ".parts")
    parts.mkdir(exist_ok=True)
    prefix = args.cache_dir / (name + ".prefix.part")

    def spans():
        result = []
        if prefix.exists() and prefix.stat().st_size:
            result.append((0, prefix.stat().st_size, prefix))
        for path in parts.glob("*.part"):
            match = re.fullmatch(r"(\d+)-(\d+)\.part", path.name)
            if not match:
                raise RuntimeError("Unexpected saved range filename")
            start, end = map(int, match.groups())
            if not 0 <= start < end <= size or path.stat().st_size != end - start:
                raise RuntimeError("Invalid saved range; preserve and inspect")
            result.append((start, end, path))
        return sorted(result)

    gaps, cursor = [], 0
    for start, end, _ in spans():
        if start < cursor or end > size:
            raise RuntimeError("Saved spans overlap or exceed the locked wheel")
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < size:
        gaps.append((cursor, size))
    chunk_size = 1024 * 1024
    missing = [(s, min(s + chunk_size, b)) for a, b in gaps for s in range(a, b, chunk_size)]
    print(f"{name}: {sum(b-a for a,b in missing)} missing bytes in {len(missing)} ranges", flush=True)

    def fetch(span):
        start, end = span
        error = None
        for attempt in range(8):
            try:
                request = urllib.request.Request(url, headers={
                    "Range": f"bytes={start}-{end-1}", "Accept-Encoding": "identity",
                })
                with urllib.request.urlopen(request, timeout=20) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end-1}/{size}":
                        raise RuntimeError("Server did not return the exact requested range")
                    data = response.read(end - start + 1)
                if len(data) != end - start:
                    raise RuntimeError("Truncated or oversized range")
                (parts / f"{start}-{end}.part").write_bytes(data)
                return len(data)
            except (OSError, RuntimeError, http.client.HTTPException) as exc:
                error = exc
                time.sleep(min(5, attempt + 1))
        raise RuntimeError(f"Range {start}-{end} failed after bounded retries: {error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, span) for span in missing]
        completed = 0
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            completed += future.result()
            if index % 20 == 0 or index == len(missing):
                print(f"ranges {index}/{len(missing)}, received {completed / 1024 / 1024:.1f} MiB", flush=True)
    cursor = 0
    with final.open("xb") as target:
        for start, end, path in spans():
            if start != cursor:
                raise RuntimeError("Incomplete assembly coverage")
            with path.open("rb") as source:
                for data in iter(lambda: source.read(chunk_size), b""):
                    target.write(data)
            cursor = end
    if cursor != size or final.stat().st_size != size or digest_file(final) != sha:
        raise RuntimeError("Final wheel length or SHA-256 mismatch; do not install")
    print(f"VERIFIED {name} {sha}", flush=True)


if __name__ == "__main__":
    main()
