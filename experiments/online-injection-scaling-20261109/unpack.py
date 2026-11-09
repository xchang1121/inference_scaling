"""Extract a retrieval-scaling package into a new directory and verify its hashes."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    manifest = json.loads((package/"MANIFEST.json").read_text(encoding="utf-8"))
    archive = package/manifest["archive"]["path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive"]["sha256"]
    assert not args.output.exists(), "choose a new extraction directory"
    output = args.output.resolve()
    output.mkdir(parents=True)
    expected = {member["path"]: member for member in manifest["members"]}
    seen = set()
    with tarfile.open(archive) as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name in expected and member.name not in seen
            data = tar.extractfile(member).read()
            entry = expected[member.name]
            assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
            target = output/Path(*name.parts)
            target.resolve().relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            seen.add(member.name)
    assert seen == set(expected)
    print(json.dumps({"verified_members": len(seen), "output": str(output)}))


if __name__ == "__main__":
    main()
