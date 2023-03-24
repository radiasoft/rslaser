# -*- coding: utf-8 -*-
"""Tests for instantiation of LaserPulse
and LaserPulseSlice
"""
from __future__ import absolute_import, division, print_function
from pykern.pkdebug import pkdp, pkdlog
from pykern.pkcollections import PKDict
import pykern.pksubprocess
import pykern.pkunit
import pykern.pkio
import re
import array
import pytest
import copy
import srwlib
from rslaser_new.pulse import pulse
from rslaser_new.optics.wavefront import (
    WavefrontSensor,
    InvalidWaveFrontSensorInputError,
)


EPSILON = 1e-2
WFR_ATTRS_LIST = [
    "arElecPropMatr",
    "arEx",
    "arEy",
    "arMomX",
    "arMomY",
    "arWfrAuxData",
    "avgPhotEn",
    "dRx",
    "dRy",
    "presCA",
    "presFT",
    "unitElFld",
    "unitElFldAng",
    "xc",
    "yc",
]


def test_instantiation01():
    WavefrontSensor("w1", 2.0)
    with pykern.pkunit.pkexcept(InvalidWaveFrontSensorInputError):
        WavefrontSensor("w1", "2.0")


def test_propagation01():
    from pykern import pkunit
    from pykern import pkio

    data_dir = pkunit.data_dir()
    work_dir = pkunit.empty_work_dir()
    p = pulse.LaserPulse()
    w = WavefrontSensor("w1", 2.0)
    r = w.propagate(p)
    actual = PKDict()
    for a in WFR_ATTRS_LIST:
        v = getattr(r, a)
        actual.update({a: v})
    pkunit.file_eq(
        expect_path=data_dir.join("res.ndiff"),
        actual=str(actual),
    )


def test_propagation02():
    p = pulse.LaserPulse()
    wfs = WavefrontSensor("w1", 2.0)
    if not type(wfs.propagate(p)) == srwlib.SRWLWfr:
        pykern.pkunit.pkfail(
            f"WavefrontSensor.propagate failed to return type {srwlib.SRWLWfr}"
        )
    p = pulse.LaserPulseSlice(0)
    with pykern.pkunit.pkexcept(InvalidWaveFrontSensorInputError):
        wfs.propagate(p)
