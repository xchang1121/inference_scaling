"""Verify and restore the complete offline analysis workspace (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


ROOT = Path(__file__).resolve().parent


def relative_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError(f'Unsafe archive path: {name!r}')
    return Path(*path.parts)


def check_file(path, expected):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')


def verify(manifest):
    for entry in manifest['direct_files']:
        check_file(ROOT / relative_path(entry['path']), entry)
    count = 0
    for archive in manifest['archives']:
        path = ROOT / relative_path(archive['path'])
        check_file(path, archive)
        expected = {entry['path']: entry for entry in archive['files']}
        if len(expected) != len(archive['files']):
            raise ValueError(f'Duplicate manifest members: {path}')
        seen = set()
        with tarfile.open(path, 'r:gz') as bundle:
            for member in bundle:
                relative_path(member.name)
                if not member.isfile() or member.name in seen or member.name not in expected:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                source = bundle.extractfile(member)
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
                entry = expected[member.name]
                if member.size != entry['bytes'] or digest != entry['sha256']:
                    raise ValueError(f'Archive member checksum mismatch: {member.name}')
                seen.add(member.name)
            if seen != set(expected):
                raise ValueError(f'Archive has missing members: {path}')
        count += len(seen)
    return count


def unpack(manifest, destination):
    destination = destination.resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError('Output must be a new or empty directory.')
    destination.mkdir(parents=True, exist_ok=True)
    for archive in manifest['archives']:
        prefix = relative_path(archive['destination'])
        with tarfile.open(ROOT / relative_path(archive['path']), 'r:gz') as bundle:
            for member in bundle:
                target = destination / prefix / relative_path(member.name)
                target.resolve().relative_to(destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--output', type=Path, help='New or empty directory to restore.')
    args = parser.parse_args()
    if args.verify_only == (args.output is not None):
        parser.error('Choose exactly one of --verify-only or --output DIR.')
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 1:
        raise ValueError('Unsupported bundle manifest schema.')
    count = verify(manifest)
    result = {'verified_archive_members': count, 'verified_direct_files': len(manifest['direct_files'])}
    if args.output is not None:
        result['restored_workspace'] = str(unpack(manifest, args.output))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
