"""Base class for input validation
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
from pykern.pkcollections import PKDict


class ValidatorBase:
    """
    Base class for LaserPulse LaserPulseSlice
    Used for input validation
    """

    def _get_params(self, params):
        if params == None:
            return self._DEFAULTS.copy()
        self._validate_type(params, PKDict, "params")
        for k in self._DEFAULTS:
            if k not in params:
                params[k] = self._DEFAULTS[k]
        return params

    def _validate_params(self, input_params):
        for p in input_params:
            if p not in self._DEFAULTS:
                raise self._INPUT_ERROR(
                    f"invalid inputs: {p} is not a parameter to {self.__class__}"
                )

    def _validate_type(self, input, target_type, params_name):
        if type(input) != target_type:
            raise self._INPUT_ERROR(
                f"invalid input type: {self.__class__} takes {params_name} as type:{target_type} for input."
            )
