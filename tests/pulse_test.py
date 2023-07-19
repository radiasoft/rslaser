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
from rslaser.pulse import pulse
import scipy
import rslaser
import srwlib


_PACKAGE_DATA_DIR = rslaser.pkg_resources.resource_filename("rslaser", "package_data")


def pulse_instantiation_test(pulse, field):
    for s in pulse.slice:
        if getattr(s, field) != getattr(pulse, field):
            pykern.pkunit.pkfail(
                f"LaserPulseSlice has different {field} than pulse as a whole"
            )


def test_sirepo_compatability():
    """
    Test of compatability between rslaser and sirepo silas master branch

    NOTE:
        If this test fails, then alert software team
    indicates rslaser interface is not compatable with
    sirepo silas master branch
    """
    from rslaser.optics.lens import Lens_srw, Drift_srw
    from rslaser.optics.crystal import Crystal

    def interpolate_across_slice(length, nslice, values):
        return scipy.interpolate.splev(
            (length / nslice) * (np.arange(nslice) + 0.5),
            scipy.interpolate.splrep(np.linspace(0, length, len(values)), values),
        ).tolist()

    try:
        p = pulse.LaserPulse(
            params=PKDict(
                nslice=3,
                num_sig_long=3.0,
                num_sig_trans=6.0,
                nx_slice=64,
                photon_e_ev=1.5,
                poltype=1,
                pulseE=0.001,
                sigx_waist=0.001,
                sigy_waist=0.001,
                tau_fwhm=2.3586500000000002e-11,
                phase_flatten_cutoff=0.85,
                tau_0=2.3586500000000002e-11,
            ),
        )

        # POSIT: mirror and watchpoints dont need
        # to be checked since they are sirepo elements
        # not rslaser elements
        elements = [
            (Drift_srw(3), ["default"]),
            (
                Crystal(
                    params=PKDict(
                        l_scale=0.001,
                        length=0.02,
                        n0=interpolate_across_slice(
                            0.02, 10, [1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
                        ),
                        n2=interpolate_across_slice(
                            0.02, 10, [30.5, 18.3, 10.4, 5.9, 3.3, 1.9]
                        ),
                        nslice=10,
                        A=0.99736924,
                        B=0.0141972275,
                        C=-0.260693,
                        D=0.99892682,
                        pop_inversion_n_cells=64,
                        pop_inversion_mesh_extent=0.01,
                        pop_inversion_crystal_alpha=120,
                        pop_inversion_pump_waist=0.00164,
                        pop_inversion_pump_wavelength=5.32e-07,
                        pop_inversion_pump_gaussian_order=2,
                        pop_inversion_pump_energy=0.0211,
                        pop_inversion_pump_type="dual",
                    ),
                ),
                ["n0n2_srw", True, False],
                80,
                "gaussian",
            ),
            (Drift_srw(0.5), ["default"]),
            (Lens_srw(2), ["default"]),
            (Drift_srw(0.25), ["default"]),
            (Drift_srw(0.25), ["default"]),
        ]

        beamline = [0, 1, 2, 3, 4, 5]
        crystal_count = 0
        for idx in beamline:
            e = elements[idx]
            if isinstance(e[0], Crystal):
                e[0].calc_n0n2(set_n=True, mesh_density=e[2], heat_load=e[3])
            p = e[0].propagate(p, *e[1])
    except Exception:
        raise AssertionError(
            """
    If this test fails, then alert software team
    indicates rslaser interface is not compatable with
    sirepo silas master branch
        """
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
        ccd=pkio.py_path(_PACKAGE_DATA_DIR).join("/20220218/photon_count_pump_off.txt"),
        wfs=pkio.py_path(_PACKAGE_DATA_DIR).join("/20220218/phase_pump_off.txt"),
        meta=pkio.py_path(_PACKAGE_DATA_DIR).join("/20220218/meta_data.dat"),
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
        ndiff_epsilon=1e-5,
    )
