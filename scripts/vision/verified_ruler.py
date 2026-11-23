"""Conservative veto for scan lines that leave the sheet at the entry silhouette."""
import numpy as np
from scipy import ndimage

from ruler_readout import classify as base_classify, inks


def classify(rgb, depth):
    result = base_classify(rgb, depth)
    if result['decision'] == 'UNKNOWN':
        return result
    lab = inks(rgb)
    y, x = np.where(lab >= 0)
    lab = lab[y.min():y.max()+1, x.min():x.max()+1]
    joined = []
    for j in result['open_rows']:
        contact = (lab == j) & ndimage.binary_dilation(lab == j+13, np.ones((3, 3), bool))
        if contact.any():
            joined.append(j)
    if not joined:
        return result
    return dict(result, decision='UNKNOWN', reason='visible bank contact contradicts claimed open row',
                contact_rows=joined)
