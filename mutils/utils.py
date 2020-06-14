import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hcipy
import scipy.optimize as spopt


def zeropad(arr, *, Q=None, N=None):
    """
    Pad an array with symmetric zeros along each dimension.  If Q is specified, the input will be
    padded by an integer multiplicative factor.  If N is specified, the input will be padded out
    to a fixed size.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    Q : int
        Integer factor by which to increase the size of the array, i.e. output.shape[0] /
        input.shape[0] = Q
    N : int
        Size of output array, i.e. output.shape[0] = N

    Returns
    -------
    np.ndarray
        Input array with equal amounts of zeros on either side along each dimension.  Size is
        increased by factor of q along each dimension relative to input.
    """
    dim = len(arr.shape)

    M = arr.shape[0]

    # Neither or both are specified
    if (Q is None and N is None) or (Q is not None and N is not None):
        raise ValueError('Either Q or N, but not both, must be specified.')

    # Q is specified but N is not
    elif Q is not None and N is None:  # Pad the current array out by an integer multiple of M
        pad_list = dim * [[M * (Q - 1) // 2, M * (Q - 1) // 2]]

    # N is specified but Q is not
    else:  # Pad the current array out to a specified width
        pad = int((N - M) // 2)  # Pad evenly on both sides so that the total width is N
        pad_list = dim * [[pad, pad]]

    return np.pad(arr, pad_list, mode='constant', constant_values=0)


def bin(arr, binfac):
    M, N = arr.shape
    new_shape = (M // binfac, binfac, N // binfac, binfac)
    return arr.reshape(new_shape).mean(axis=3).mean(axis=1)


def crop(arr, desired_shape):
    """
    Slice out the centermost pixels of an array along each dimension.

    Parameters
    ----------
    arr : array_like
        Input array
    desired_shape : tuple of int
        The number of pixels to include along each dimension.

    Returns
    -------
    np.ndarray
        Central M x M region of input.

    """
    if any(np.array(desired_shape) > np.array(arr.shape)):
        raise ValueError('Sliced array must be smaller than input array')

    slices = []
    for shape, desired in zip(arr.shape, desired_shape):
        lower = int((shape - desired) // 2)
        upper = int((shape + desired) // 2)
        slices.append(np.s_[lower:upper])

    return arr[slices[0], slices[1]]


def vector_to_image(vector, mask):
    """
    Put a 1D vector of dark-zone pixel values (e.g., estimated fields or intensities) into a 2D
    image, with evetything outside the dark zone masked out.

    Parameters
    ----------
    vector : array_like
    dark_hole : array_like
        2D dark zone boolean array

    Returns
    -------
    array_like
        Masked array

    """
    image = np.zeros(mask.shape, dtype=vector.dtype)
    image[mask] = vector
    return np.ma.masked_where(mask == 0, image)


def compute_color_limit(*arrs):
    """
    Find the maximum magnitude within any array passed in as argument.
    Used to obtain colorbar limits for display purposes.

    Parameters
    ----------
    arrs : list of array_like

    Returns
    -------
    float

    """
    return np.max([np.abs(arr).max() for arr in arrs])


def add_colorbar(fig, ax, im, label='', size='5%', pad=0.05, visible=True):
    """
    Create a colorbar whose height is matched its plot, with a desired fractional width.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure handle, i.e. from plt.subplots()
    ax : matplotlib.axes.Axes
        Axis handle, i.e. from plt.subplots()
    im : matplotlib.image.AxesImage
        Image handle, i.e. from ax.imshow()
    label : str
        Label for colorbar
    size : str
        Colorbar width as a percentage of axis width
    pad : float
        Fractional padding between image and colorbar
    visible: bool
        Whether to actually generate and display colorbar after modifying axis to make room for it.
        An example case is when we want multiple adjacent figures to share a colorbar.  All of them
        have to be modified to have room for individual colorbars, but only the last one is
        actually displayed.
    Returns
    -------
    None

    """
    divider = make_axes_locatable(ax)

    cax = divider.append_axes('right', size=size, pad=pad)
    cb = fig.colorbar(im, cax=cax, label=label)

    if not visible:
        cb.remove()

    return cb


def rms(arr):
    """
    Compute the root-mean-square value for an array.

    Parameters
    ----------
    arr : np.ndarray
        Input array

    Returns
    -------
    float
    """
    return np.ma.sqrt(np.ma.mean(np.square(arr)))


def normalize_rms(arr, target_rms, mask=None):
    """
    Normalize an array to have a specified root-mean-square (RMS) statistic.

    Parameters
    ----------
    arr : array_like
        Input array
    target_rms : float
        Desired RMS value of the output array
    mask : array_like
        Binary mask indicating the elements of the input array over which to compute the RMS.  All
        elements where the mask has a value of zero are ignored in the input array.  Must have the
        same dimensions as the input.

    Returns
    -------
    array_like
        Input array, scaled so that rms(arr) = target_rms
    """

    if mask is not None:
        actual_rms = rms(np.ma.masked_where(mask, arr, copy=True))
    else:
        actual_rms = rms(arr)

    return arr * target_rms / actual_rms


def circleshade(axis, radius):
    """
    Generate a circular aperture with antialiased edges using the ramp algorithm described in [1].

    Parameters
    ----------
    axis : array_like
        Coordinate axis.  Assumed to be the same in both directions.
    radius : float
        Aperture radius.

    Returns
    -------
    array_like
        Antialiased aperture

    References
    ----------
    [1] Will and Fienup, "Algorithm for exact area-weighted antialiasing of discrete circular
        apertures," JOSA A 37, 688-696 (2020).
    """
    rg = radial_grid(axis)  # Radial grid
    step = axis[1] - axis[0]
    delta_r = 0.5 + (radius - rg) / step
    return np.minimum(np.maximum(delta_r, 0), 1)


def radial_grid(axis):
    """
    Compute a memory-efficient radial grid using array broadcasting.

    Parameters
    ----------
    axis : np.ndarray
        1D coordinate axis

    Returns
    -------
    np.ndarray
        2D grid with radial coordinates generated by axis
    """
    x, y = broadcast(axis)
    return np.sqrt(x ** 2 + y ** 2)


def broadcast(axis):
    """
    Use numpy array broadcasting to return two views of the input axis that behave like a row
    vector (x) and a column vector (y), and which can be used to build memory-efficient
    coordinate grids without using meshgrid.

    Given an axis with length N, the naive approach using meshgrid requires building two NxN arrays,
    and combining them (e.g. to obtain a radial coordinate grid) produces a third NxN array.
    Using array broadcasting, one only needs to create two separate views into the original Nx1
    vector to create the final NxN array.

    Parameters
    ----------
    axis : np.ndarray
        1D coordinate axis

    Returns
    -------
    tuple of np.ndarray
        Two views into axis that behave like a row and column vector, respectively

    """
    x = axis[None, :]
    y = axis[:, None]
    return x, y


def extent_from_axis(axis):
    """
    Return appropriate values for the extent keyword of matplotlib.pyplot.imshow().  The values
    are computed from a coordinate axis such that when an image is displayed with matplotlib,
    the upper and lower limits are consistent with the view that each pixel represents a square
    region of space with side length (step).  This function assumes that the image is square with
    the same pixel spacing and pixel count along the horizontal and vertical directions.

    Parameters
    ----------
    axis : array_like
        Coordinate axis

    Returns
    -------
    tuple of float
        Tuple in the form (lower_x, upper_x, lower_y, upper_y) whose entries specify the physical
        coordinates of the upper and lower limits of the displayed image in the horizontal and
        vertical directions.
    """
    step = axis[1] - axis[0]
    lower = axis.min() - 0.5 * step
    upper = axis.max() + 0.5 * step

    return 2 * (lower, upper)


def hcipy_grid_from_axes(*axes):
    """
    Create an hcipy coordinate grid from a linear axis.  Assumes that both dimensions have the same
    extent and sample spacing.

    Note that this ignores the extra arguments 'center' and 'has_center' in
    hcipy.make_uniform_grid(), because this is intended to be used for display purposes only; e.g.


    Parameters
    ----------
    axis : array_like
        Coordinate axis
    """
    steps = [axis[1] - axis[0] for axis in axes]
    extents = [axis.max() - axis.min() + step for step, axis in zip(steps, axes)]
    return hcipy.make_uniform_grid([len(axis) for axis in axes], extents)


def field_from_array(arr, *axes):
    grid = hcipy_grid_from_axes(*axes)
    return hcipy.Field(arr.ravel(), grid)

# TODO: implement array_from_field


def check_gradient(obj, x0):
    """
    Check analytical gradient.  This differs from the scipy.optimize function in two ways.  First, the objective is assumed to return BOTH the objective AND the gradient,
    because I'm using algorithmic differentiation.  Second, the normalized sum-of-squared differences metric is computed instead of just SSD, which is much more useful.
    """

    # Generate internal functions that extract just the objective and gradient
    def gradient(u):
        _, grad = obj(u)
        return grad

    def objective(u):
        J, _ = obj(u, evaluate_gradient=False)
        return J

    grad_analytical = gradient(x0)
    grad_numerical = spopt.approx_fprime(x0, objective, epsilon=1e-6)

    return np.linalg.norm(grad_analytical - grad_numerical) / np.linalg.norm(grad_numerical)


def embed(arr, mask, mask_output=False):
    """
    Embed an array into a higher-dimensional array using a boolean mask.

    Parameters
    ----------
    arr : array_like
    mask : array_like
    mask_output : bool
        Whether to convert the output to a masked array.

    Returns
    -------
    array_like

    """
    output = np.zeros(mask.shape, dtype=np.complex128)
    output[mask] = arr

    if mask_output:
        output = np.ma.masked_where(mask == 0, output)

    return output
