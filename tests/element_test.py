u"""Tests for Crystal and CrystalSlice
"""
from pykern.pkdebug import pkdp, pkdlog
from pykern.pkcollections import PKDict
import pykern.pkunit
import pytest
from rslaser_new.optics import element, lens, drift, crystal
from rslaser_new.pulse import pulse
import srwlib


def test_instantiation01():
    crystal.Crystal()
    with pykern.pkunit.pkexcept(element.ElementException):
        crystal.Crystal('fail')
    with pykern.pkunit.pkexcept(element.ElementException):
        crystal.Crystal(PKDict(slice_params=PKDict()))
    c = crystal.Crystal()
    for s in c.slice:
        if s.length != c.length/c.nslice:
            pykern.pkunit.pkfail('CrystalSlice had length not equal to Crystal wrapper length/nslice')


def test_crystal_nslice():
    crystal.Crystal(PKDict(nslice=51))
    crystal.Crystal(PKDict(n0=[1], n2=[1]))
    with pykern.pkunit.pkexcept(element.ElementException, "you've specified"):
        crystal.Crystal(PKDict(nslice=51, n0=[1], n2=[1]))


def crystal_slice_prop_test(prop_type):
    c = crystal.CrystalSlice()
    p = pulse.LaserPulse()
    p = c.propagate(p, prop_type)
    if type(p) != pulse.LaserPulse:
        pykern.pkunit.pkfail('Crystal slice propagaition failed to return LaserPulse type')


def test_instantiation02():
    c = crystal.CrystalSlice()


def test_propagation():
    with pykern.pkunit.pkexcept(element.ElementException):
        crystal_slice_prop_test('default')
    with pykern.pkunit.pkexcept(NotImplementedError):
        crystal_slice_prop_test('attenuate')
    with pykern.pkunit.pkexcept(NotImplementedError):
        crystal_slice_prop_test('placeholder')
    c = crystal.CrystalSlice()
    with pykern.pkunit.pkexcept(KeyError):
        c.propagate(pulse.LaserPulse(), 'should raise')


def test_prop_with_gain():
    data_dir = pykern.pkunit.data_dir()

    def _prop(prop_type):
        c = crystal.Crystal(
            PKDict(
                n2=[16],
                l_scale=0.001,
            )
        )
        p = pulse.LaserPulse(
            PKDict(
                nx_slice = 32,
                ny_slice = 32,
            )
        )
        c.propagate(p, prop_type, calc_gain=True)
        w = p.slice_wfr(0)
        i = srwlib.array('f', [0]*w.mesh.nx*w.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(i, w, 6, 0, 3, w.mesh.eStart, 0, 0)
        pykern.pkunit.file_eq(
            data_dir.join(prop_type+"_intensity.ndiff"),
            actual=str(i),
        )

    for prop_type in ("n0n2_srw", "n0n2_lct", "gain_calc"):
        _prop(prop_type)


def test_instantiation03():
    drift.Drift(0.01)


def test_propagation05():
    d = drift.Drift(0.01)
    p = pulse.LaserPulse()
    d.propagate(p)
    trigger_prop_fail(
        drift.Drift(0.01).propagate,
        pulse.LaserPulse()
        )


def test_instantiation04():
    lens.Lens(0.2)


def test_propagation06():
    l = lens.Lens(0.2)
    l.propagate(pulse.LaserPulse())
    trigger_prop_fail(
        lens.Lens(0.01).propagate,
        pulse.LaserPulse()
        )

def trigger_prop_fail(prop_func, pulse):
    with pykern.pkunit.pkexcept(
        element.ElementException,
        'Invalid element="should raise" should have raised'
        ):
        prop_func(pulse, 'should raise')
