"""Frozen model counterexample: can the visible triangle set certify the structural stop label?

Not a second live trajectory. No change is made to the benchmark scene or captured frame.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/task'))
from ground_truth import GroundTruth, NU, NV


def digest(data):
    return hashlib.sha256(data).hexdigest()


def surface(constraints):
    alive = {tuple(sorted((a, b))) for a, b, flag in constraints if flag}
    faces = []
    for j in range(NV-1):
        for i in range(NU-1):
            a = j*NU+i
            for face in ((a, a+1, a+NU+1), (a, a+NU+1, a+NU)):
                if all(tuple(sorted((face[k-1], face[k]))) in alive for k in range(3)):
                    faces.append((j, *face))
    return np.array(faces, dtype='<i4')


def length(constraints):
    def call(name):
        if name != 'tissueConstraints':
            raise ValueError(name)
        return constraints
    gt = GroundTruth(SimpleNamespace(sheet_origin=True, call=call))
    gt.strict = True
    with gt.purpose('scoring'):
        return gt.cut_length()


def audit(out):
    run_path = ROOT / 'validation/task/m5b_v1_9_1_marked_truth.json'
    run = json.loads(run_path.read_bytes())
    model = ROOT / 'src/tissue/tissue.lua'
    frame = ROOT / run['config']['perception']['dump_dir'] / 'tick_016.npz'
    if (out.exists() or run['final_state'] != 'DONE' or run['trace'][15]['tick'] != 16
            or digest((ROOT / 'scripts/task/ground_truth.py').read_bytes()) != run['source_sha256']['scripts/task/ground_truth.py']):
        raise ValueError('output exists or reference mismatch')
    with np.load(frame, allow_pickle=False) as f:
        before = f['tissue_alive']
        original = length(before)
        if not original == f['cut_length'] == run['trace'][15]['cut_length_m'] == .01:
            raise ValueError('capture is not the structural endpoint')
        xyz = digest(f['nodes'].tobytes())
    faces = surface(before)
    examples = []
    for index, (a, b, flag) in enumerate(before):
        a, b = sorted((int(a), int(b)))
        if flag or b-a != NU-1 or a//NU == b//NU:
            continue
        other = before.copy()
        other[index, 2] = 1
        alternative = length(other)
        if alternative >= run['config']['cut']['length_m']:
            continue
        if not np.array_equal(faces, surface(other)):
            raise ValueError('not a render-invariant perturbation')
        examples.append(dict(restored_constraint=[a, b], alternative_length_m=alternative))
    if not examples:
        raise ValueError('no ambiguous frozen labels found')
    result = dict(scope='Frozen fixed-vertex counterfactual; no claim that alternative dynamics reach the same pose.',
                  reference_sha256=digest(run_path.read_bytes()), frame_sha256=digest(frame.read_bytes()),
                  tissue_lua_sha256=digest(model.read_bytes()), visual_triangles=int(len(faces)),
                  node_sha256=xyz, material_row_and_face_sha256=digest(faces.tobytes()),
                  measured_length_m=original, examples=examples)
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(original, 'm;', len(examples), 'non-rendered single-spring changes cross the stop threshold')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
