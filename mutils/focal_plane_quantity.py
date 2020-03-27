import numpy as np


class FocalPlaneQuantity:
    """
    Class that assists in converting values between various focal-plane units of interest in
    astronomy, such as arcseconds, radians, diffraction widths (wavelength over aperture), and
    meters (spatial coordinates in the paraxial focal plane of a lens).

    Unit conversion can be modeled as a graph, with each connected pair of nodes representing one
    "step" of conversion:

                        Diffraction widths -> radians -> arcsec
                                                      -> meters

    In this implementation, quantities can be specified in any of the above units via setter
    properties, which update the internal values for the quantity in other units.  Internally, to
    simplify logic, when a value is set in some unit, it is automatically converted to radians.  A
    single function that converts radians to all other units (including the input units) can then be
    reused throughout.

    For handling of more general units, see the astropy.units package.  Astropy.units does not
    appear to support the "parameterized" conversions between equivalent quantities performed here.
    For example, arcseconds and meters are generally not equivalent units- however, according to
    paraxial Fourier optics, angular units, spatial frequency, and transverse coordinates are
    related by

                fx (spatial frequency) = x (length) / (wavelength * focal length)

    where terms can be rearranged to obtain angles, lengths, or spatial frequencies given a
    wavelength and focal length of a lens.

    Additionally, it is sometimes useful to represent focal-plane quantities in units of wavelength
    over aperture ("diffraction widths").  The first null of the far-field diffraction pattern from
    a circular aperture with diameter D, at wavelength lambda, is located at the angle

                                    theta = 1.22 lambda / D

    where theta is in radians.  This angle specifies the diffraction limit for an aperture with
    diameter D.  Thus, to describe angular quantities as multiples of the diffraction limit, we can
    adopt the "diffraction widths" pseudo-unit, for which the above quantity would simply be given
    as
                                        theta = 1.22
    """

    # Conversion factor from radians to arcseconds.  Constant- doesn't depend on any of the object
    # attributes such as focal length, pupil diameter, or wavelength
    RAD_TO_ARCSEC = (180 / np.pi) * 3600

    def __init__(self, focal_length, diameter, lambda_0):
        self.focal_length = focal_length
        self.diameter = diameter
        self.lambda_0 = lambda_0
        self.LAMD_TO_RAD = self.lambda_0 / self.diameter
        self.RAD_TO_M = self.focal_length
        self._lamD = None
        self._rad = None
        self._arcsec = None
        self._m = None

    @property
    def rad(self):
        return self._rad

    @rad.setter
    def rad(self, val):
        self._rad = val
        self._arcsec = self._rad * self.RAD_TO_ARCSEC
        self._lamD = self._rad / self.LAMD_TO_RAD
        self._m = self._rad * self.RAD_TO_M

    @property
    def lamD(self):
        return self._lamD

    @lamD.setter
    def lamD(self, val):
        self.rad = val * self.LAMD_TO_RAD
        assert np.allclose(val, self._lamD)

    @property
    def arcsec(self):
        return self._arcsec

    @arcsec.setter
    def arcsec(self, val):
        self.rad = val / self.RAD_TO_ARCSEC
        assert np.allclose(val, self._arcsec)

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, val):
        self.rad = val / self.RAD_TO_M
        assert np.allclose(val, self._m)

    def print_all(self):
        output = \
            f"""
            theta_x [rad]        = {self.rad}
            theta_x [arcsec]     = {self.arcsec}
            theta_x [lambda_0/D] = {self.lamD}
                  x [m]          = {self.m}
            """
        print(output)


if __name__ == '__main__':
    focal_length = 20  # Focal length [m]
    lambda_0 = 1600e-9  # Wavelength [m]
    diameter = 8.  # Diameter [m]

    converter = FocalPlaneQuantity(focal_length, diameter, lambda_0)
    converter.lamD = 1.

    converter.print_all()
