from srwlib import SRWLOptC, SRWLOptL
from . import Element


class Lens(Element):
    """
    Create lens element

    Args:
        f (float): focal length [m]

    Returns:
        SRW beamline element representing lens
    """

    def __init__(self, f):
        self.length = 0
        self._srwc = SRWLOptC(
            [SRWLOptL(f, f)], [[0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]]
        )
