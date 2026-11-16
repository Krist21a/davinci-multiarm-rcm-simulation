"""Shared helpers for the reproducible endoscope scene build.

All paths are resolved relative to the repository. The Remote API host/port are
command-line options, so a build never attaches to an unrelated CoppeliaSim
instance by accident.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_SCENE = REPO / 'scenes' / 'davinci_foot_docking_rcm.ttt'
ENDOSCOPE_SCENE = REPO / 'scenes' / 'davinci_endoscope_task.ttt'
CONFIG = REPO / 'config'
ENDO_CONFIG = CONFIG / 'endoscope'
ENDO_SRC = REPO / 'src' / 'endoscope'
ENDO_VALIDATION = REPO / 'validation' / 'endoscope'
URDF = REPO / 'models' / 'davinci_foot_docking.urdf'
MESHES = REPO / 'models' / 'meshes'


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=23000,
                        help='ZeroMQ Remote API rpcPort of a CoppeliaSim instance reserved for this build')
    parser.add_argument('--stage-dir', type=Path, default=Path(tempfile.gettempdir()),
                        help='ASCII-only directory used to stage .ttt files for loadScene/saveScene')
    parser.add_argument('--timeout', type=float, default=180.0, help='Remote API receive timeout in seconds')


def _import_client():
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        return RemoteAPIClient
    except ImportError:
        root = os.environ.get('COPPELIASIM_ROOT')
        if root:
            sys.path.insert(0, str(Path(root) / 'programming' / 'zmqRemoteApi' / 'clients' / 'python' / 'src'))
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
            return RemoteAPIClient
        raise SystemExit('coppeliasim-zmqremoteapi-client is not installed; '
                         'pip install -r requirements-build.txt or set COPPELIASIM_ROOT')


def connect(args):
    RemoteAPIClient = _import_client()
    client = RemoteAPIClient(host=args.host, port=args.port)
    client.socket.RCVTIMEO = int(args.timeout * 1000)
    sim = client.require('sim')
    return client, sim


def stage_dir(args) -> Path:
    path = Path(args.stage_dir).resolve()
    # CoppeliaSim 4.10 does not round-trip non-ASCII paths through the Remote API
    # (loadScene/getStringParam block), so scene files are staged in an ASCII folder.
    if not str(path).isascii():
        raise SystemExit(f'--stage-dir must be an ASCII-only path, got {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def stop_simulation(client, sim, timeout=10.0):
    if sim.getSimulationState() == sim.simulation_stopped:
        return
    sim.stopSimulation()
    t0 = time.time()
    while sim.getSimulationState() != sim.simulation_stopped:
        if time.time() - t0 > timeout:
            raise RuntimeError('simulation did not stop')
        try:
            client.step()
        except Exception:
            time.sleep(0.05)


def load_scene(client, sim, args, scene: Path, name: str) -> Path:
    stop_simulation(client, sim)
    staged = stage_dir(args) / name
    shutil.copy2(scene, staged)
    sim.loadScene(str(staged))
    time.sleep(0.3)
    return staged


def save_scene(sim, args, target: Path):
    staged = stage_dir(args) / target.name
    sim.saveScene(str(staged))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(staged, target)


def aliases(sim):
    return {sim.getObjectAlias(h): h for h in sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)}


def lua(value):
    """Serialise JSON-compatible data as a Lua table literal (same format as the original builder)."""
    if isinstance(value, dict):
        return '{' + ','.join('[' + json.dumps(k) + ']=' + lua(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '{' + ','.join(lua(v) for v in value) + '}'
    return json.dumps(value)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _json_default(value):
    # numpy scalars/arrays (np.bool_, np.float64, ndarray) -> plain Python values
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def write_json(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform (.gitattributes normalises *.json to LF; recorded SHA-256 values must match the checkout).
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + '\n')


def read_text(path: Path) -> str:
    # Lua sources are committed with LF endings (.gitattributes); normalise for comparisons.
    return Path(path).read_text(encoding='utf-8').replace('\r\n', '\n')


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return Path(path).resolve().relative_to(REPO).as_posix()


def controller_source(rcm_config, collision_pairs) -> str:
    """The exact text embedded as /RCMController: generated data + committed controller source."""
    return ('CONFIG=' + lua(rcm_config) + '\nCOLLISION_PAIRS=' + lua(collision_pairs) + '\n'
            + read_text(ENDO_SRC / 'rcm_controller.lua'))


def panel_data(endoscope_cfg, rcm_config):
    """Data table prepended to the panel script, derived from endoscope.json and the RCM config."""
    arms = []
    for a in rcm_config:
        is_ecm = a['arm'] == 'ecm'
        arms.append({'label': 'ECM (endoscope)' if is_ecm else a['arm'].upper(),
                     'short': 'ECM' if is_ecm else 'RCM ' + a['arm'][3:],
                     'shaft': '/' + a['shaft_frame'], 'reference': '/' + a['arm'] + '_RCM_reference',
                     'yaw_deg': a['limits']['yaw_deg'], 'pitch_deg': a['limits']['pitch_deg'],
                     'insertion_min': a['limits']['insertion_min'], 'insertion_max': a['limits']['insertion_max'],
                     'insertion_home': a['insertion_home']})
    return {'camera': '/' + endoscope_cfg['camera']['alias'], 'floating_view': endoscope_cfg['panel']['floating_view'],
            'arms': arms}


def panel_source(data) -> str:
    return 'ENDOSCOPE=' + lua(data) + '\n' + read_text(ENDO_SRC / 'rcm_panel.lua')


def matrix_to_np(m):
    import numpy as np
    return np.array(m, dtype=float).reshape(3, 4)


def np_to_matrix(rot, pos):
    import numpy as np
    return np.column_stack((rot, pos)).flatten().tolist()
