from srwlib import SRWLOptC, SRWLOptL, SRWLOptD
from . import Element
import numpy as np
from pykern.pkcollections import PKDict


class Lens_srw(Element):
    """
    Create lens element

    Args:
        f (float): focal length [m]

    Returns:
        SRW beamline element representing lens
    """

    def __init__(self, f):
        self.length = 0
        self.prop_type = "srw"
        self._srwc = SRWLOptC(
            [SRWLOptL(f, f)], [[0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]]
        )


class Lens_lct(Element):
    """
    Create lens element

    Args:
        f (float): focal length [m]
        l_scale (float): scaling factor for LCT calculations

    Returns:
        ABCD_LCT beamline element representing lens
    """

    def __init__(self, f, l_scale=1.0):
        self.length = 0
        self.prop_type = "lct"
        self.focal_length = f
        self.l_scale = l_scale

        d1 = np.array([[1.0, self.length], [0.0, 1.0]])
        l1 = np.array([[1.0, 0.0], [-1.0 / self.focal_length, 1.0]])
        lens_abcd = np.matmul(l1, d1)
        self.abcd_matrix = PKDict(
            A=lens_abcd[0, 0],
            B=lens_abcd[0, 1],
            C=lens_abcd[1, 0],
            D=lens_abcd[1, 1],
        )


class Beamsplitter(Element):
    def __init__(self, transmitted_fraction):
        self.transmitted_fraction = transmitted_fraction
        self.prop_type = "beamsplitter"

        if (self.transmitted_fraction < 0.0) or (self.transmitted_fraction > 1.0):
            raise ElementException(
                f"Invalid transmitted fraction passed to beamsplitter"
            )


class Drift_srw(Element):
    def __init__(self, length):
        self.length = length
        self.prop_type = "srw"
        self._srwc = SRWLOptC(
            [SRWLOptD(self.length)],
            [[0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]],
        )


class Drift_lct(Element):
    def __init__(self, length, l_scale=1.0):
        self.length = length
        self.prop_type = "lct"
        self.l_scale = l_scale
        drift_abcd = np.array([[1.0, self.length], [0.0, 1.0]])
        self.abcd_matrix = PKDict(
            A=drift_abcd[0, 0],
            B=drift_abcd[0, 1],
            C=drift_abcd[1, 0],
            D=drift_abcd[1, 1],
        )


class Telescope_lct(Element):
    def __init__(self, f1, f2, d1, d2, d3, l_scale=1.0):
        self.prop_type = "lct"
        self.focal_length_1 = f1
        self.focal_length_2 = f2
        self.drift_length_1 = d1
        self.drift_length_2 = d2
        self.drift_length_3 = d3
        self.l_scale = l_scale

        mat_lens_1 = np.array([[1.0, 0.0], [-1.0 / self.focal_length_1, 1.0]])
        mat_lens_2 = np.array([[1.0, 0.0], [-1.0 / self.focal_length_2, 1.0]])
        mat_drift_1 = np.array([[1.0, self.drift_length_1], [0.0, 1.0]])
        mat_drift_2 = np.array([[1.0, self.drift_length_2], [0.0, 1.0]])
        mat_drift_3 = np.array([[1.0, self.drift_length_3], [0.0, 1.0]])

        telescope_abcd = np.matmul(
            mat_drift_3,
            np.matmul(
                mat_lens_2, np.matmul(mat_drift_2, np.matmul(mat_lens_1, mat_drift_1))
            ),
        )

        self.abcd_matrix = PKDict(
            A=telescope_abcd[0, 0],
            B=telescope_abcd[0, 1],
            C=telescope_abcd[1, 0],
            D=telescope_abcd[1, 1],
        )
