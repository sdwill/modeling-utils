import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_axis(N, step, coords='pc'):
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
    coords : str
        Either 'pc' (pixel-centered) or 'ipc' (inter-pixel-centered)

    Returns
    -------
    Union[np.ndarray, np.matrix]
        The output coordinate axis
    """
    axis = np.arange(-N // 2, N // 2, dtype=np.float64) * step

    if coords == 'ipc':
        axis += 0.5 * step

    return axis


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


def circleshade(size, *, clear_size=None, radius=None, center=(0, 0)):
    """
    Render a circular aperture with shaded gray-level pixels.  Originally implemented in wfscore
    package by Alden Jurling, Matthew Bergkoetter et al.

    Parameters
    -----------
    size : int
        Array size for created pupil (will be square).
    clear_size : float
        Diameter in pixels for shaded circle.
    radius : float
        Radius in pixels for shaded circle.
    center : 2-tuple
        Center point (w.r.t to center of array) for shaded circle. Defaults to
        ``(0, 0)``

    Returns
    --------
    circle : np.ndarray
        Shaded circle.
    """
    if clear_size is None and radius is not None:
        clear_size = radius * 2
    elif clear_size is None and radius is None:
        raise ValueError('Radius or diameter must be specificed')
    X, Y = np.mgrid[:size, :size] - size // 2
    # Center is defined relative to row/col with DC shift.
    R = ((X - center[0]) ** 2 + (Y - center[1]) ** 2) ** 0.5
    radph = clear_size / 2 + 0.5
    circ_tmp = radph - R
    circ = np.minimum(np.maximum(circ_tmp, 0), 1)
    return circ
