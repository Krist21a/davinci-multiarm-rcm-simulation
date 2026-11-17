"""Download the da Vinci Si cannula models from jhu-dvrk/dvrk-cannulas (MIT) into models/meshes/cannula/dvrk/.

Pinned to one commit; SOURCE.json records URL, SHA-256 and size of every file. The OpenSCAD source is kept:
it documents the geometry (8.7 mm bore, 125 mm tube, RCM marked by a cross hole 95 mm below the tube top).
"""
import hashlib
import json
import sys
import urllib.request

from simtools import MESHES, relative, write_json

COMMIT = '4603d0a9f22bc21b4c44c4a3bbfc4a39b98fff74'
REPO = 'jhu-dvrk/dvrk-cannulas'
OUT = MESHES / 'cannula' / 'dvrk'
FILES = ['LICENSE', 'Si/README.md', 'Si/Si-cannula.scad', 'Si/Si-cannula-8mm.stl', 'Si/Si-cannula-12mm.stl']


def main():
    manifest = OUT / 'SOURCE.json'
    old = json.loads(manifest.read_text(encoding='utf-8'))['files'] if manifest.exists() else {}
    files = {}
    for path in FILES:
        target = OUT / path
        if target.exists() and path in old and hashlib.sha256(target.read_bytes()).hexdigest() == old[path]['sha256']:
            files[path] = old[path]
            continue
        url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path}'
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if data.startswith(b'version https://git-lfs'):
            sys.exit(f'{path}: got a Git LFS pointer instead of the file')
        if not path.endswith('.stl'):
            data = data.replace(b'\r\n', b'\n')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[path] = dict(url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        print('fetched', relative(target), len(data), flush=True)
    write_json(manifest, {'_note': 'Downloaded by scripts/build/fetch_dvrk_cannulas.py. MIT licence, see LICENSE.',
                          'repository': f'https://github.com/{REPO}', 'commit': COMMIT, 'files': files})


if __name__ == '__main__':
    main()
