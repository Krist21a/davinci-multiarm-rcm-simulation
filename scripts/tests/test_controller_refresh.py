"""Check stopped-scene source refresh without starting CoppeliaSim."""
import hashlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'build'))
from simtools import connect, refresh_controller, refreshed_controller_source  # noqa: E402

MARKER = '-- Software RCM controller for the endoscope task scene\nOLD CODE'
PREFIX = 'CONFIG={}\nCOLLISION_PAIRS={}\n'


class ControllerRefresh(unittest.TestCase):
    def test_connection_failure_has_bounded_socket_shutdown(self):
        client = Mock()
        client.require.side_effect = TimeoutError('unreachable')
        args = Mock(host='localhost', port=23271, timeout=3)
        with patch('simtools._import_client', return_value=Mock(return_value=client)):
            with self.assertRaises(TimeoutError):
                connect(args)
        self.assertEqual(client.socket.LINGER, 0)
        self.assertEqual(client.socket.SNDTIMEO, 3000)
        self.assertEqual(client.socket.RCVTIMEO, 3000)
        client.socket.close.assert_called_once_with(linger=0)
        client.context.term.assert_called_once()

    def test_preserves_geometry_identified_data(self):
        prefix = PREFIX + 'INSTRUMENTS={unique_geometry_data=123}\n'
        code = refreshed_controller_source(prefix + MARKER, tissue=True)
        self.assertTrue(code.startswith(prefix))
        self.assertIn('function tissueGrasp', code)
        self.assertIn('function hasInstrumentCapability', code)
        self.assertNotIn('OLD CODE', code)
        self.assertEqual(refreshed_controller_source(code, tissue=True), code)

    def test_base_scene_does_not_get_tissue_capabilities(self):
        code = refreshed_controller_source(PREFIX + MARKER)
        self.assertNotIn('function tissueGrasp', code)
        with self.assertRaises(ValueError):
            refreshed_controller_source(PREFIX + MARKER, tissue=True)

    def test_unknown_script_is_rejected(self):
        for code in ['unknown', 'CONFIG={}\n' + MARKER, PREFIX + 'BAD={}\n' + MARKER]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                refreshed_controller_source(code)

    def test_active_simulation_cannot_be_modified(self):
        sim = Mock(simulation_stopped=0)
        sim.getSimulationState.return_value = 1
        with self.assertRaises(RuntimeError):
            refresh_controller(sim, 7)
        sim.setStringProperty.assert_not_called()

    def test_readback_and_hash(self):
        sim = Mock(simulation_stopped=0)
        sim.getSimulationState.return_value = 0
        state = {'code': PREFIX + MARKER}
        sim.getStringProperty.side_effect = lambda *args: state['code']
        sim.setStringProperty.side_effect = lambda h, p, code: state.update(code=code)
        digest = refresh_controller(sim, 7)
        self.assertEqual(digest, hashlib.sha256(state['code'].encode()).hexdigest())

    def test_mismatch_restores_original_script(self):
        sim = Mock(simulation_stopped=0)
        sim.getSimulationState.return_value = 0
        old = PREFIX + MARKER
        sim.getStringProperty.side_effect = [old, 'wrong code']
        with self.assertRaises(RuntimeError):
            refresh_controller(sim, 7)
        self.assertEqual(sim.setStringProperty.call_args.args, (7, 'code', old))


if __name__ == '__main__':
    unittest.main()