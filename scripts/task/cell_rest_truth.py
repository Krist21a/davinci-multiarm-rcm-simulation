"""Isolated truth adapter dividing transverse distances by measured per-row rests.

GroundTruth.cut_line_tension once normalised every row by the uniform
H*(1-pretension); on coupons whose centre cell is not H wide that is a metrology
error. Production GroundTruth now normalises by the readback world.cutline_rest_m
with an explicit-legacy scalar fallback; this adapter keeps the registered
measured-only contract — even on the frozen legacy controller a missing or
invalid rest readback is an error, not a fallback.
"""
from surface_seam_world import SeamTruth


class CellRestTruth(SeamTruth):
    measured_rest_required = True
