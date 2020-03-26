import numpy as np
import hcipy

class CoordinateAxis(np.ndarray):
    def __new__(cls, arr, centering='pc'):
        obj = np.asarray(arr).view(cls)
        obj.centering = centering
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._ = getattr(obj, 'symmetric', None)

    @property
    def step(self):
        return self[1] - self[0]

    @property
    def extent(self):
        return self.max() - self.min() + self.step

    @property
    def display_extent(self):
        lower = self.min() - 0.5 * self.step
        upper = self.max() + 0.5 * self.step

        return lower, upper

    def extend(self):
        pass  # TODO: how to extend self in-place? Look at hcipy for guidance


def create_axis(N, step, centering='pc'):
    """
    Create a one-dimensional coordinate axis with a given size and step size.  Can be constructed
    to follow either the FFT (pixel-centered) or MFT (inter-pixel-centered) convention,
    which differ by half a pixel.

    Parameters
    ----------
    N : int
        Number of pixels in output axis
    step : float
        Physical step size between axis elements
    centering : str
        Either 'pc' (pixel-centered) or 'ipc' (inter-pixel-centered)

    Returns
    -------
    CoordinateAxis
        The output coordinate axis
    """
    axis = np.arange(-N // 2, N // 2, dtype=np.float64) * step

    if centering == 'ipc':
        axis += 0.5 * step

    return CoordinateAxis(axis, centering=centering)


if __name__ == '__main__':
    M = 4
    dx = 1
    centering = 'pc'
    axis = create_axis(M, dx, centering)

    print(axis)
    print(axis.step)
