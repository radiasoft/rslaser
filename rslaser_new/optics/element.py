from srwlib import srwl
from rslaser_new.utils.validator import ValidatorBase

class ElementException(Exception):
    pass

class Element(ValidatorBase):
    def propagate(self, laser_pulse, prop_type='default'):
        if prop_type != 'default':
            raise ElementException(f'Non default prop_type "{prop_type}" passed to propagation')
        if not hasattr(self, '_srwc'):
            raise ElementException(f'_srwc field is expected to be set on {self}')
        for w in laser_pulse.slice:
            srwl.PropagElecField(w.wfr,self._srwc)
        return laser_pulse
