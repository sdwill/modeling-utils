import numpy as np
from numpy.lib.function_base import flip
from . import utils, dfts
from .coordinate_axis import create_axis


class DeformableMirror:
    """
    Efficient model for a deformable mirror that uses the matrix Fourier transform.
    """
    def __init__(self, num_act_across, pupil_axis, diam,
                 crosstalk=0.15,
                 actuator_mask=None,
                 inclination=0.,
                 translation=0.,
                 print_through_model=None,
                 flip_x=False):
        self.num_act_across = num_act_across
        self.diam = diam
        self.inclination = inclination
        self.translation = translation
        self.obliquity = np.cos(self.inclination * np.pi / 180)
        self.flip_x = flip_x

        self.actuator_spacing_y = (diam / num_act_across)
        self.actuator_spacing_x = self.actuator_spacing_y * self.obliquity

        if actuator_mask is None:
            self.actuator_mask = np.ones((num_act_across, num_act_across), dtype=bool)
        else:
            self.actuator_mask = actuator_mask

        if print_through_model is None:
            M = len(pupil_axis)
            self.print_through_model = np.zeros((M, M), dtype=np.float64)
        else:
            self.print_through_model = print_through_model

        self.num_actuator = self.actuator_mask.sum()
        self.sigma_x = self.actuator_spacing_x / np.sqrt(-2 * np.log(crosstalk))
        self.sigma_y = self.actuator_spacing_y / np.sqrt(-2 * np.log(crosstalk))
        self.pupil_axis = pupil_axis

        # Precompute the Fourier-domain transfer function for our influence function
        self.transfer_function = dfts.fft(self.kernel) / len(pupil_axis)
        self.command = np.zeros(self.num_actuator, dtype=np.float64)

        # Fourier transforming matrices for the x and y directions
        self.Fx = np.exp(-1j * 2 * np.pi * np.outer(self.fx, self.cx))
        self.Fy = np.exp(-1j * 2 * np.pi * np.outer(self.fy, self.cy))

    @staticmethod
    def gaussian_influence_function(x, y, sigma_x, sigma_y):
        """ Circular Gaussian influence function """
        xb = x[None, :]
        yb = y[:, None]
        return np.exp(-0.5 * ((xb / sigma_x) ** 2 + (yb / sigma_y) ** 2))

    @staticmethod
    def sinc_influence_function(x, y, sigma_x, sigma_y):
        return np.sinc(x / sigma_x) * np.sinc(y / sigma_y)

    @property
    def x(self):
        return ((self.pupil_axis + self.translation) * self.obliquity)

    @property
    def y(self):
        return self.pupil_axis

    @property
    def kernel(self):
        return self.gaussian_influence_function(self.x, self.y, self.sigma_x, self.sigma_y)

    @property
    def fx(self):
        """ Spatial frequency axis along x"""
        return dfts.conjugate_axis(self.x)

    @property
    def fy(self):
        """ Spatial frequency axis along y"""
        return dfts.conjugate_axis(self.y)

    @property
    def a(self):
        """ DM actuator coordinate axis (actuator index) """
        return np.arange(-self.num_act_across / 2, self.num_act_across / 2)

    @property
    def cx(self):
        """ DM actuator center positions """
        if self.num_act_across % 2:  # Odd number of actuators
            centering = 'pc'
            shift = self.actuator_spacing_x
        else:
            centering = 'ipc'
            shift = 0

        step = self.actuator_spacing_x * (self.num_act_across - 1) / self.num_act_across
        return create_axis(self.num_act_across, step, centering) + shift + self.translation

    @property
    def cy(self):
        """ DM actuator center positions """
        if self.num_act_across % 2:  # Odd number of actuators
            centering = 'pc'
            shift = self.actuator_spacing_y
        else:
            centering = 'ipc'
            shift = 0

        step = self.actuator_spacing_y * (self.num_act_across - 1) / self.num_act_across
        return create_axis(self.num_act_across, step, centering) + shift

    @property
    def surface(self):
        """ Get the surface for the current stored set of commands """
        return self.forward(self.command)

    def forward(self, command):
        """ Get the surface for an arbitrary set of input commands """
        command = utils.embed(command, self.actuator_mask)
        ft_command = np.linalg.multi_dot([self.Fx, command, self.Fy.T])
        product = self.transfer_function * ft_command
        surface = dfts.ifft(product).real * len(self.fx) + self.print_through_model

        if self.flip_x:
            surface = surface[:, ::-1]
            
        return surface
    def reverse(self, gradient):
        if self.flip_x:
            gradient = gradient[:, ::-1]
            
        ft_gradient = dfts.fft(gradient.real) / len(self.fx)
        product = self.transfer_function.conj() * ft_gradient
        return np.linalg.multi_dot([self.Fx.conj().T, product, self.Fy.conj()])[self.actuator_mask]
