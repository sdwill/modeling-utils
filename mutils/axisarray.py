import numpy as np
from functools import reduce
from operator import add

from .coordinate_axis import CoordinateAxis
from .utils import field_from_array


class AxisArray(np.ndarray):
    def __new__(cls, arr, *axes):
        obj = np.asarray(arr).view(cls)
        obj.axes = axes
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.axes = getattr(obj, 'axes', None)

    def to_field(self):
        return field_from_array(self.ravel(), *self.axes)

    @property
    def display_extent(self):
        """ Get the display extent for all axes together """
        return reduce(add, [axis.display_extent for axis in self.axes])

if __name__ == '__main__':
    from coordinate_axis import create_axis

    M = 256
    dx = 1
    axis = create_axis(M, dx, centering='ipc')
    arr = AxisArray(np.random.rand(M), axis)

    print(arr)
    print(arr.axes)

