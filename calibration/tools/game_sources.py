"""Canonical game IDs are bound to exact video bytes, never inferred from names."""
from pathlib import Path
import json
import re

CATALOG = Path(__file__).resolve().parents[2] / 'data' / 'game-sources.json'


def read_catalog(path=CATALOG):
    value = json.loads(Path(path).read_text())
    if value.get('schema') != 'verified-game-sources-v1':
        raise ValueError('Unsupported game-source catalog')
    result = {}
    hashes = set()
    for row in value['games']:
        gid, sha = row['game_id'], row['video_sha256']
        if gid in result or sha in hashes or not re.fullmatch(r'[0-9a-f]{64}', sha):
            raise ValueError('Ambiguous or malformed game-source catalog')
        if row.get('source_identity_verified') is not True:
            raise ValueError('Game-source identity has not passed pixel verification')
        result[gid] = row
        hashes.add(sha)
    return result


def validate_bindings(sources, catalog=None):
    bound = [(key.split(':')[0], source) for key, source in sources.items()
             if re.fullmatch(r'\d+:video', key)]
    if not bound:
        return
    catalog = read_catalog() if catalog is None else catalog
    for gid, source in bound:
        if gid not in catalog or source.get('sha256') != catalog[gid]['video_sha256']:
            raise ValueError(f'Wrong video for Trace game {gid}; check the source catalog and its identity evidence. '
                             'The historical Granite Bay 4 / 13232938 pairing is invalid.')
