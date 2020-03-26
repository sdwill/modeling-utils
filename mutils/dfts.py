import numpy as np
from .utils import radial_grid


def fft(x, norm=None):
    """
    Compute the Fourier transform of signal x using an FFT, including shifting and de-shifting
    operations.

    Parameters
    ----------
    x : np.ndarray
        N-dimensional input array

    Returns
    -------
    np.ndarray
        N-dimensional output array

    """
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm=norm))


def ifft(X, norm=None):
    """
    Compute the inverse Fourier transform of signal X using an IFFT, including shifting and
    de-shifting operations.

    Parameters
    ----------
    X : np.ndarray
        N-dimensional input array

    Returns
    -------
    np.ndarray
        N-dimensional output array

    """
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X), norm=norm))


def conjugate_axis(x):
    """
    Compute the coordinate axis in the Fourier domain corresponding to the output of an FFT
    algorithm

    Parameters
    ----------
    x : np.ndarray
        Coordinate axis of function in original domain
    Returns
    -------
    np.ndarray
        Coordinate axis of function transformed by FFT

    """
    dx = np.diff(x)[0]
    N = len(x)
    Bx = np.abs(x.max() - x.min()) + dx
    dfx = 1 / Bx
    # fx_max = 1 / dx
    return np.arange(-N // 2, N // 2) * dfx


def mft(arr, axis_in, axis_out, forward=True, scaling='forward'):
    """
    Compute a matrix discrete Fourier transform of the 1D function f using the axes x and u of the
    original and conjugate domains, respectively.

    Parameters
    ----------
    arr : np.ndarray
        The two-dimensional function to be transformed (should be a column vector)
    axis_in : np.ndarray
        The coordinate axis of f, as a column vector
    axis_out : np.ndarray
        The coordinate axis of the Fourier transform of f, as a column vector
    scaling : {'forward', 'reverse', 'none'}
        How to scale the output.  'forward' and 'reverse' apply Riemann-sum scaling using the step
        size of the input axis and output axis, respectively.  'none' applies no scaling.
    Returns
    -------
    np.matrix
        Column vector containing the Fourier transform of F
    """
    sign = -1 if forward else 1
    pre = np.exp(sign * 1j * 2 * np.pi * np.outer(axis_out, axis_in))
    post = np.exp(sign * 1j * 2 * np.pi * np.outer(axis_in, axis_out))
    output = np.linalg.multi_dot([pre, arr, post])

    if scaling == 'forward':
        coeff = (axis_in[1] - axis_in[0]) ** 2
    elif scaling == 'reverse':
        coeff = (axis_out[1] - axis_out[0]) ** 2
    else:
        coeff = 1

    return coeff * output


def angular_spectrum_transfer(axis, wavelength, dz, paraxial=False):
    """
    Returns the transfer function for an angular spectrum propagation for a given wavelength and
    propagation distance.

    Parameters
    ----------
    axis : np.ndarray
        The spatial frequency axis onto which the transfer function will be built.
    wavelength : float
        Center wavelength of propagation [m]
    dz : float
        Propagation distance [m]
    paraxial: bool
        Whether to use the exact or paraxial (double-DFT Fresnel) transfer function

    Returns
    -------
    np.ndarray

    """
    rho = radial_grid(axis)  # Radial spatial frequency grid

    if paraxial:
        return np.exp(-1j * np.pi * wavelength * dz * (rho ** 2))  # Paraxial transfer function
    else:
        kz = np.sqrt(1 - (wavelength * rho) ** 2)  # z-component of k-vector of propagated field
        return np.exp(1j * 2 * np.pi * (dz / wavelength) * kz)  # Exact transfer function


def fft_angular_spectrum(arr, axis, wavelength, dz, paraxial=False):
    """
    Perform an angular spectrum propagation over distance dz, using FFTs.

    Parameters
    ----------
    arr : array_like
        Input field
    axis : array_like
        The 1D coordinate axis for both dimensions of the input field
    wavelength : float
        Propagation wavelength
    dz : float
        Propagation distance
    paraxial : bool, optional
        Whether to use the exact (False) or paraxial (True) angular spectrum transfer function.
        Defaults to False.

    Returns
    -------
    array_like
        Input field propagated over distance dz
    """
    fx = conjugate_axis(axis)
    H = angular_spectrum_transfer(fx, wavelength, dz, paraxial=paraxial)
    return ifft(H * fft(arr))

