"""Tests for Crystal and CrystalSlice
"""
from pykern.pkdebug import pkdp, pkdlog
from pykern.pkcollections import PKDict
import pykern.pkunit
import pytest
from rslaser.optics import element, lens, drift, crystal
from rslaser.pulse import pulse
from scipy import constants
import srwlib
import numpy


def test_instantiation01():
    crystal.Crystal()
    with pykern.pkunit.pkexcept(element.ElementException):
        crystal.Crystal("fail")
    with pykern.pkunit.pkexcept(element.ElementException):
        crystal.Crystal(PKDict(slice_params=PKDict()))
    c = crystal.Crystal()
    for s in c.slice:
        if s.length != c.length / c.nslice:
            pykern.pkunit.pkfail(
                "CrystalSlice had length not equal to Crystal wrapper length/nslice"
            )
    crystal.Crystal(
        PKDict(
            nslice=10,
            pop_inversion_n_cells=32,
        )
    )


def test_crystal_nl_kick():
    data_dir = pykern.pkunit.data_dir()
    p = pulse.LaserPulse(
        PKDict(
            photon_e_ev=1.23984198 / (799e-9 * 1e6),
            nslice=1,
            nx_slice=80,
            pulseE=1.0e-6,
            tau_fwhm=(4 * numpy.pi * (1.64e-3 / 2) ** 2 / 799e-9) / constants.c * 2.355,
            sigx_waist=1.64e-3 / 2,
            sigy_waist=1.64e-3 / 2,
            num_sig_trans=6,
        )
    )
    c = crystal.Crystal(
        PKDict(
            length=0.025,
            nslice=1,
            n0=[1.76],
            n2=[12.151572393382056],
            delta_n_array=numpy.load(str(data_dir.join("delta_narray.npy"))),
            delta_n_mesh_extent=numpy.max(
                numpy.loadtxt(data_dir.join("radpts.txt"), delimiter=",")
            ),
            l_scale=1e-3,
            A=0.99736924,
            B=1.41972275 / 1e2,
            C=-0.00260693 * 1e2,
            D=0.99892682,
        )
    )
    c.propagate(p, "n0n2_srw", nl_kick=True)
    c.propagate(p, "n0n2_srw")


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
        pykern.pkunit.pkfail(
            "Crystal slice propagaition failed to return LaserPulse type"
        )


def test_instantiation02():
    c = crystal.CrystalSlice()


def test_propagation():
    with pykern.pkunit.pkexcept(element.ElementException):
        crystal_slice_prop_test("default")
    with pykern.pkunit.pkexcept(NotImplementedError):
        crystal_slice_prop_test("attenuate")
    with pykern.pkunit.pkexcept(NotImplementedError):
        crystal_slice_prop_test("placeholder")
    c = crystal.CrystalSlice()
    with pykern.pkunit.pkexcept(KeyError):
        c.propagate(pulse.LaserPulse(), "should raise")


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
                nx_slice=32,
            )
        )
        c.propagate(p, prop_type, calc_gain=True)
        w = p.slice_wfr(0)
        i = srwlib.array("f", [0] * w.mesh.nx * w.mesh.ny)
        srwlib.srwl.CalcIntFromElecField(i, w, 6, 0, 3, w.mesh.eStart, 0, 0)
        pykern.pkunit.file_eq(
            data_dir.join(prop_type + "_intensity.ndiff"),
            actual=str(i),
        )

    for prop_type in ("n0n2_srw", "n0n2_lct", "gain_calc"):
        _prop(prop_type)


def test_instantiation03():
    lens.Drift_srw(0.01)


def test_propagation05():
    d = lens.Drift_srw(0.01)
    p = pulse.LaserPulse()
    d.propagate(p)
    trigger_prop_fail(lens.Drift_srw(0.01).propagate, pulse.LaserPulse())


def test_instantiation04():
    lens.Lens_srw(0.2)


def test_propagation06():
    l = lens.Lens_srw(0.2)
    l.propagate(pulse.LaserPulse())
    trigger_prop_fail(lens.Lens_srw(0.01).propagate, pulse.LaserPulse())


def trigger_prop_fail(prop_func, pulse):
    with pykern.pkunit.pkexcept(
        element.ElementException, 'Invalid element="should raise" should have raised'
    ):
        prop_func(pulse, "should raise")
