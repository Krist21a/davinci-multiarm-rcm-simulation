"""Opt-in wide central-cell seam world (registered separately; the real boundary moved).

The only change vs the verified slide chain is the flat-grid u-coordinate expression:
outer cells compress to 9 mm per side so the single central cell is 5 mm wide, moving
the actual host-material boundary. Removed nodes are never inferred -- every node and
triangle still exists. Guide projection, controller pins and truth-only full mode are
inherited unchanged.
"""
from pathlib import Path
from unittest.mock import patch

from seam_slide_world import SeamSlideWorld, staged_source as slide_source

ROOT = Path(__file__).resolve().parents[2]

_NEEDLE = '(i-(nu-1)/2)*h'
_ANCHOR = 'vmul(spec.u,(i-(nu-1)/2)*h)'
_WIDE = '((i<=11 and (-11.5+i*(9/11))) or (2.5+(i-12)*(9/11)))*h'


def wide_source(parent):
    """Slide-staged source with the flat-grid u coordinate widened and the passive
    certificate print appended. Raises rather than guessing if the base text drifts."""
    base = slide_source(parent)
    if base.count(_NEEDLE) == 1:
        base = base.replace(_NEEDLE, _WIDE)
    elif base.count(_ANCHOR) == 1:
        base = base.replace(_ANCHOR, 'vmul(spec.u,' + _WIDE + ')')
    else:
        raise RuntimeError('ambiguous flat-grid definition')
    return base + '\n' + (ROOT / 'src/tissue/cell_certificate_print.lua').read_text(encoding='utf-8')


class WideCellWorld(SeamSlideWorld):
    def start(self):
        with patch('seam_slide_world.staged_source', wide_source):
            super().start()
