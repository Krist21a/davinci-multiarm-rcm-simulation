"""Download the da Vinci S/Si instrument meshes used by config/instruments/ from jhu-dvrk/dvrk_model.

The files are pinned to one dvrk_model commit and written under models/meshes/instrument/dvrk/ with the
upstream MIT licence and a SOURCE.json manifest (commit, URL, SHA-256 and size of every file). Meshes are
stored with Git LFS upstream, so they are read from media.githubusercontent.com.

Re-running is safe: a file whose SHA-256 already matches the manifest is not downloaded again.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from simtools import MESHES, relative, write_json

COMMIT = '90472a933995acaa88d84eb5c58aeee4cd17e9f3'
REPO = 'jhu-dvrk/dvrk_model'
OUT = MESHES / 'instrument' / 'dvrk'

# Stages of the Si instruments in config/instruments/ (dvrk_model urdf/common/instruments/instruments.yaml).
MESH_FILES = [
    'meshes/instruments/wrist_pitch/0091/wrist_pitch_0091.obj',
    'meshes/instruments/wrist_yaw/0091/wrist_yaw_0091.obj',
    'meshes/instruments/wrist_pitch/0107/wrist_pitch_0107.obj',
    'meshes/instruments/wrist_yaw/0107/wrist_yaw_0107.obj',
    'meshes/instruments/tip/006/tip_006_1.obj',
    'meshes/instruments/tip/006/tip_006_2.obj',
    'meshes/instruments/tip/179/tip_179_1.obj',
    'meshes/instruments/tip/179/tip_179_2.obj',
    'meshes/instruments/tip/172/tip_172_1.obj',
    'meshes/instruments/tip/172/tip_172_2.obj',
    'meshes/instruments/tip/093/tip_093_1.obj',
    'meshes/instruments/tip/093/tip_093_2.obj',
]
TEXT_FILES = [
    'LICENSE',
    'urdf/common/instruments/instruments.yaml',
    'urdf/common/instruments/wrist_pitch/wrist_pitch_0091.urdf.xacro',
    'urdf/common/instruments/wrist_yaw/wrist_yaw_0091.urdf.xacro',
    'urdf/common/instruments/wrist_pitch/wrist_pitch_0107.urdf.xacro',
    'urdf/common/instruments/wrist_yaw/wrist_yaw_0107.urdf.xacro',
    'urdf/common/instruments/tip/tip_006.urdf.xacro',
    'urdf/common/instruments/tip/tip_179.urdf.xacro',
    'urdf/common/instruments/tip/tip_172.urdf.xacro',
    'urdf/common/instruments/tip/tip_093.urdf.xacro',
]


def url_for(path):
    host = 'media.githubusercontent.com/media' if path.startswith('meshes/') else 'raw.githubusercontent.com'
    return f'https://{host}/{REPO}/{COMMIT}/{path}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args()
    manifest_path = OUT / 'SOURCE.json'
    old = json.loads(manifest_path.read_text(encoding='utf-8'))['files'] if manifest_path.exists() else {}
    files = {}
    for path in TEXT_FILES + MESH_FILES:
        target = OUT / path
        if target.exists() and path in old and hashlib.sha256(target.read_bytes()).hexdigest() == old[path]['sha256']:
            files[path] = old[path]
            continue
        url = url_for(path)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if data.startswith(b'version https://git-lfs'):
            sys.exit(f'{path}: got a Git LFS pointer instead of the file')
        if path.endswith('.obj') or path.endswith('.xacro') or path == 'LICENSE' or path.endswith('.yaml'):
            data = data.replace(b'\r\n', b'\n')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[path] = dict(url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        print('fetched', relative(target), len(data), flush=True)
    write_json(manifest_path, {
        '_note': 'Downloaded by scripts/build/fetch_dvrk_instruments.py. MIT licence, see LICENSE in this folder.',
        'repository': f'https://github.com/{REPO}', 'commit': COMMIT, 'files': files})


if __name__ == '__main__':
    main()
