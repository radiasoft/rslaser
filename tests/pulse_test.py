# -*- coding: utf-8 -*-
"""Tests for instantiation of LaserPulse and LaserPulseSlice
"""
from __future__ import absolute_import, division, print_function
import math
import numpy as np
from pykern.pkdebug import pkdp, pkdlog
from pykern.pkcollections import PKDict
import pykern.pkunit
import pytest
from rslaser_new.pulse import pulse
import scipy
import rslaser_new
import srwlib


_PACKAGE_DATA_DIR = rslaser_new.pkg_resources.resource_filename(
    "rslaser_new", "package_data"
)


def pulse_instantiation_test(pulse, field):
    for s in pulse.slice:
        if getattr(s, field) != getattr(pulse, field):
            pykern.pkunit.pkfail(
                f"LaserPulseSlice has different {field} than pulse as a whole"
            )


def test_instantiation():
    pulse.LaserPulse()
    p = pulse._LASER_PULSE_DEFAULTS.copy()
    p.update(PKDict(blonk=9))
    with pykern.pkunit.pkexcept(pulse.InvalidLaserPulseInputError):
        pulse.LaserPulse(p)
    with pykern.pkunit.pkexcept(pulse.InvalidLaserPulseInputError):
        pulse.LaserPulse(PKDict(foo="bar", hello="world"))


# TODO (gurhar1133): propagation is a work in progress.
# def test_cavity_propagation():
#     from pykern import pkunit
#     from pykern import pkio

#     data_dir = pkunit.data_dir()
#     work_dir = pkunit.empty_work_dir()
#     L_cav = 8
#     dfL = 1
#     dfR = 1

#     L_cryst = 2 * 1e-2
#     n0 = 2
#     n2 = 0.02

#     wavefrontEnergy = 1.55
#     lam = (
#         scipy.constants.c
#         * scipy.constants.value("Planck constant in eV/Hz")
#         / wavefrontEnergy
#     )

#     L_eff = L_cav + (1 / n0 - 1) * L_cryst
#     beta0 = math.sqrt(L_eff * (L_cav / 4 + dfL) - L_eff**2 / 4)
#     sigx0 = math.sqrt(lam * beta0 / 4 / math.pi)

#     lc = laser_cavity.LaserCavity(
#         PKDict(
#             drift_right_length=L_cav / 2 - L_cryst / 2,
#             drift_left_length=L_cav / 2 - L_cryst / 2,
#             lens_left_focal_length=L_cav / 4 + dfR,
#             lens_right_focal_length=L_cav / 4 + dfL,
#             n0=n0,
#             n2=n2,
#             L_half_cryst=L_cryst / 2,
#             pulse_params=PKDict(
#                 phE=wavefrontEnergy,
#                 nslice=11,
#                 slice_params=PKDict(**pulse._LASER_PULSE_SLICE_DEFAULTS),
#             ),
#         )
#     )

#     results = []

#     def intensity_callback(position, vals):
#         (x, y) = lc.laser_pulse.rmsvals()
#         results.append(
#             [
#                 lc.laser_pulse.pulsePos(),
#                 x,
#                 y,
#                 lc.laser_pulse.intensity_vals(),
#                 lc.laser_pulse.energyvals(),
#             ]
#         )

#     lc.propagate(num_cycles=4, callback=intensity_callback)

#     ndiff_files(
#         data_dir.join("res.txt"),
#         pkio.write_text(
#             work_dir.join("res_actual.txt"),
#             str(results[-1]),
#         ),
#         work_dir.join("ndiff.out"),
#         data_dir,
#     )


def test_from_file():
    from pykern import pkunit
    from pykern import pkio

    data_dir = pkunit.data_dir()
    pulse_inputs = pulse._LASER_PULSE_DEFAULTS.copy()
    pulse_inputs.nslice = 1
    f = PKDict(
        ccd=pkio.py_path(_PACKAGE_DATA_DIR).join("ccd_pump_off.txt"),
        wfs=pkio.py_path(_PACKAGE_DATA_DIR).join("wfs_pump_off.txt"),
        meta=pkio.py_path(_PACKAGE_DATA_DIR).join("wfs_meta.dat"),
    )
    wavefront = pulse.LaserPulse(
        pulse_inputs,
        files=f,
    ).slice_wfr(0)
    intensity = srwlib.array("f", [0] * wavefront.mesh.nx * wavefront.mesh.ny)
    srwlib.srwl.CalcIntFromElecField(
        intensity, wavefront, 6, 0, 3, wavefront.mesh.eStart, 0, 0
    )
    pkunit.file_eq(
        data_dir.join("2d_wf_intensity.ndiff"),
        actual=str(intensity),
        ndiff_epsilon=1e-10,
    )
