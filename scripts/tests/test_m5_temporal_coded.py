import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
with patch.object(sys, 'path', [str(ROOT / 'scripts/learn'), *sys.path]):
    from m5_temporal_coded import (INTACT_ROW, OPEN_ROW, PRESENT, ROWS, UNKNOWN,
                                  Exposure, TemporalEvidence, read_exposure)


def exposure(tick, n, *, hidden_prefix=0, optics='wide', marker=True, local=None, overrides=()):
    rows = [OPEN_ROW] * n + [INTACT_ROW] * (ROWS - n)
    rows[:hidden_prefix] = [(UNKNOWN, UNKNOWN)] * hidden_prefix
    for i, row in overrides:
        rows[i] = row
    return Exposure(tick, optics, tuple(rows), local, marker)


def approach():
    evidence = TemporalEvidence(exposure(0, 0))
    for seq, (start, end) in enumerate(((0, 3), (3, 6), (6, 8)), 1):
        before = exposure(2 * seq, start, hidden_prefix=max(start - 1, 0))
        after = exposure(2 * seq + 1, end, hidden_prefix=max(start - 1, 0))
        assert evidence.record_pair(seq, before, after) == end - start
    return evidence


class TemporalCodedTests(unittest.TestCase):
    def test_hidden_entry_and_later_narrow_confirmation(self):
        e = approach()
        self.assertEqual(e.prefix, 8)
        self.assertTrue(e.wants_endpoint_look)
        look = exposure(9, 8, hidden_prefix=7, optics='narrow', local=8)
        self.assertEqual(e.review_look(7, look, jaws_closed=True, motion_checks_ok=True), 'BELOW')
        self.assertEqual(e.record_pair(4, exposure(10, 8, hidden_prefix=7),
                                       exposure(11, 9, hidden_prefix=7)), 1)
        look = exposure(14, 9, hidden_prefix=8, optics='narrow', local=9)
        self.assertEqual(e.review_look(11, look, jaws_closed=True, motion_checks_ok=True), 'TARGET_MATCH')
        self.assertFalse(e.wants_endpoint_look)
        self.assertEqual(e.review_look(11, look, jaws_closed=True, motion_checks_ok=True), 'UNKNOWN')

    def test_missing_or_contradictory_bite_cannot_be_backfilled(self):
        cases = (
            (5, exposure(10, 8), exposure(11, 9)),
            (4, exposure(10, 8), exposure(12, 9)),
            (4, exposure(10, 8), exposure(11, 9, marker=False)),
            (4, exposure(10, 8), exposure(11, 9, hidden_prefix=8)),
            (4, exposure(10, 8), exposure(11, 9, overrides=((4, (PRESENT, UNKNOWN)),))),
            (4, exposure(10, 8), exposure(11, 9, overrides=((9, (UNKNOWN, UNKNOWN)),))),
            (4, exposure(10, 8, overrides=((9, (UNKNOWN, UNKNOWN)),)), exposure(11, 10)),
        )
        for seq, before, after in cases:
            with self.subTest(seq=seq, before=before, after=after):
                e = approach()
                self.assertIsNone(e.record_pair(seq, before, after))
                self.assertIsNone(e.record_pair(4, exposure(10, 8), exposure(11, 9)))
                self.assertFalse(e.wants_endpoint_look)

    def test_zero_growth_is_not_advance_and_overcut_is_not_target(self):
        e = approach()
        self.assertEqual(e.record_pair(4, exposure(10, 8), exposure(11, 8)), 0)
        self.assertEqual(e.record_pair(5, exposure(12, 8), exposure(13, 11)), 3)
        self.assertEqual(e.prefix, 11)
        self.assertFalse(e.wants_endpoint_look)
        self.assertEqual(e.review_look(13, exposure(14, 9, optics='narrow', local=9),
                                       jaws_closed=True, motion_checks_ok=True), 'UNKNOWN')

    def test_fails_closed_on_mismatched_or_unusable_look(self):
        correct = exposure(12, 9, optics='narrow', local=9)
        cases = (
            (8, correct, True, True),
            (9, exposure(9, 9, optics='narrow', local=9), True, True),
            (9, exposure(12, 9, local=9), True, True),
            (9, exposure(12, 9, optics='narrow', local=9, marker=False), True, True),
            (9, exposure(12, 8, optics='narrow', local=8), True, True),
            (9, correct, False, True),
            (9, correct, True, False),
        )
        for origin, look, closed, guarded in cases:
            e = approach()
            self.assertEqual(e.record_pair(4, exposure(8, 8), exposure(9, 9)), 1)
            self.assertEqual(e.review_look(origin, look, jaws_closed=closed, motion_checks_ok=guarded), 'UNKNOWN')
            self.assertFalse(e.wants_endpoint_look)

    def test_unknown_loaded_seed_and_unsupported_optics_refuse(self):
        e = TemporalEvidence(exposure(0, 0, overrides=((0, (UNKNOWN, UNKNOWN)),)))
        self.assertIsNone(e.record_pair(1, exposure(2, 0), exposure(3, 3)))
        img, depth = np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.float32)
        self.assertIsNone(read_exposure(2, img, depth, focal=460., far=2., tcp_local=(0, 0, 0)).optics)


if __name__ == '__main__':
    unittest.main()
