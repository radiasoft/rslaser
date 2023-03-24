# -*- coding: utf-8 -*-
"""Tests for LaserPulseEnvelope
"""
from __future__ import absolute_import, division, print_function
import math
import numpy as np
from pykern.pkdebug import pkdp, pkdlog
from pykern.pkcollections import PKDict
import pykern.pkunit
import pytest
from rslaser_new.pulse import pulse
import scipy.constants as const


def test_instantiation():
    pulse.LaserPulseEnvelope()
    with pykern.pkunit.pkexcept(pulse.InvalidLaserPulseInputError):
        pulse.LaserPulseEnvelope(1)
    with pykern.pkunit.pkexcept(pulse.InvalidLaserPulseInputError):
        pulse.LaserPulseEnvelope(PKDict(test="test"))


def test_eval_ex():
    e = pulse.LaserPulseEnvelope()
    e.evaluate_envelope_ex(np.random.rand(12), np.random.rand(12), 0.1)
    e.evaluate_envelope_ex(1, 2, 3)
    with pykern.pkunit.pkexcept(TypeError):
        e.evaluate_envelope_ex("should fail")
