import numpy as np
from . import utils, dfts


class DeformableMirror:
    def __init__(self, num_act_across, x, diam):
        self.num_act_across = num_act_across
        self.diam = diam
        self.sigma = 1 / (num_act_across / 2)  # Width parameter

        # Coordinate axes
        # Spatial coordinates in DM plane
        self.x = x

        # Precompute the Fourier-domain transfer function for our influence function
        self.transfer_function = dfts.fft(self.kernel, norm='ortho')
        self.command = np.zeros((num_act_across, num_act_across), dtype=np.float64)
        self.fourier_matrix = np.exp(-1j * 2 * np.pi * np.outer(self.c, self.fx))

    @staticmethod
    def gaussian_influence_function(x, y, sigma):
        """ Circular Gaussian influence function """
        return np.exp(-(x ** 2 + y ** 2) / (0.5 * sigma) ** 2)

    @property
    def kernel(self):
        xb, yb = utils.broadcast(self.x)
        return self.gaussian_influence_function(xb, yb, self.sigma)

    @property
    def fx(self):
        """ Spatial frequency axis """
        return dfts.conjugate_axis(self.x)

    @property
    def a(self):
        """ DM actuator coordinate axis (actuator index) """
        return np.arange(-self.num_act_across / 2, self.num_act_across / 2)

    @property
    def c(self):
        """ DM actuator center positions """
        return 0.5 * self.sigma * (self.a + 0.5)

    @property
    def surface(self):
        """ Get the surface for the current stored set of commands """
        return self.forward(self.command)

    def forward(self, command):
        """ Get the surface for an arbitrary set of input commands """
        ft_command = np.linalg.multi_dot([self.fourier_matrix.T, command, self.fourier_matrix])
        product = self.transfer_function * ft_command
        return dfts.ifft(product, norm='ortho').real

    def reverse(self, gradient):
        ft_gradient = dfts.fft(gradient, norm='ortho')
        product = self.transfer_function.conj() * ft_gradient
        return np.linalg.multi_dot(
            [self.fourier_matrix.conj(), product, self.fourier_matrix.conj().T]).real