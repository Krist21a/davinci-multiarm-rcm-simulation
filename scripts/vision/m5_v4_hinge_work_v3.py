"""Same-frame planar recovery only for missing early hinge work charts."""
import m5_v4_hinge_work_v1 as primary
import m5_v4_hinge_work_v2 as planar
from m5_v4_hinge_work_v1 import margin


def observe(rgb,z,f,far,sigma,prefix,goal=9):
    args = (rgb,z,f,far,sigma,prefix,goal)
    result = primary.chart(*args)
    if result is not None or type(prefix) is not int or not 0<=prefix<6:
        return result,'primary'
    return planar.chart(*args),'early_planar'


def chart(rgb,z,f,far,sigma,prefix,goal=9):
    return observe(rgb,z,f,far,sigma,prefix,goal)[0]
