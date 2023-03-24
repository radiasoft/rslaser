from srwlib import SRWLOptC, SRWLOptD
from . import Element


class Drift(Element):
    def __init__(self, length):
        self.length = length
        self._srwc = SRWLOptC(
            [SRWLOptD(self.length)],
            [[0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]],
        )
