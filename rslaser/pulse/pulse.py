# -*- coding: utf-8 -*-
"""Definition of a laser pulse
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import array
import math
import cmath
import copy
import numpy as np
from pykern.pkcollections import PKDict
from numpy.polynomial.hermite import hermval
import rslaser.optics.wavefront as rswf
import rsmath.const as rsc
import rslaser.utils.unit_conversion as units
import rslaser.utils.srwl_uti_data as srwutil
import scipy.constants as const
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.ndimage.filters import gaussian_filter
from scipy import special
from skimage.restoration import unwrap_phase
import scipy.optimize as opt
import srwlib
from srwlib import srwl
from rslaser.utils.validator import ValidatorBase
import matplotlib.pyplot as plt

_LASER_PULSE_DEFAULTS = PKDict(
    nslice=3,
    pulse_direction=0.0,  # 0 corresponds to 'right' or 'z' or 'forward',
    photon_e_ev=1.5,
    num_sig_long=3.0,
    dist_waist=0,
    tau_fwhm=0.1 / const.c / math.sqrt(2.0),  # also tau_c: the chirped pulse length
    tau_0=0.0,
    pulseE=0.001,
    sigx_waist=1.0e-3,
    sigy_waist=1.0e-3,
    num_sig_trans=6,
    nx_slice=64,
    poltype=1,
    mx=0,
    my=0,
    phase_flatten_cutoff=0.85,
    d_lambda_RMS=0.0,
)

_ENVELOPE_DEFAULTS = PKDict(
    w0=0.1,
    a0=0.01,
    dw0x=0.0,
    dw0y=0.0,
    dzwx=0.0,
    dzwy=0.0,
    z_center=0,
    x_shift=0.0,
    y_shift=0.0,
    photon_e_ev=1.5,
    tau_fwhm=0.1 / const.c / math.sqrt(2.0),
)


class InvalidLaserPulseInputError(Exception):
    pass


class LaserPulse(ValidatorBase):
    """
    The LaserPulse contains an array of LaserPulseSlice instances, which track
    details of the evolution in time.

    Assumes a longitudinal gaussian profile when initializing

    Args:
        params (PKDict):
                photon_e_ev (float): Photon energy [eV]
                nslice (int): number of slices
                chirp (float): energy variation from first to last slice in laser pulse [eV]
                dist_waist (float): distance from waist at which initial wavefront is calculated [m]
                w0 (float): beamsize of laser pulse at waist [m]
                a0 (float): laser amplitude, a=0.85e-9 lambda[micron] sqrt(I[W/cm^2])
                dw0x (float): horizontal variation in waist size [m]
                dw0y (float): vertical variation in waist size [m]
                dzwx (float): location (in z) of horizontal waist [m]
                dzwy (float): location (in z) of vertical waist [m]
                tau_fwhm (float): FWHM laser pulse length [s] (this is electric profile) "stretched pulse duration"
                tau_0 (float): Fourier-limited duration [s] (this is electric profile)
                pulseE (float): total laser pulse energy [J]
                z_center (float): # longitudinal location of pulse center [m]
                x_shift (float): horizontal shift of the spot center [m]
                y_shift (float): vertical shift of the spot center [m]
                sigx_waist (float): horizontal RMS waist size [m]
                sigy_waist (float): vertical RMS waist size [m]
                nx_slice (int): no. of horizontal mesh points in slice
                num_sig_trans (int): no. of sigmas for transverse Gsn range
                pulseE (float): maximum pulse energy for SRW Gaussian wavefronts [J]
                poltype (int): polarization 1- lin. hor., 2- lin. vert., 3- lin. 45 deg., 4- lin.135 deg., 5- circ. right, 6- circ. left
                mx (int): transverse Gauss-Hermite mode order in horizontal direction
                my (int): transverse Gauss-Hermite mode order in vertical direction

    Returns:
        instance of class with attributes:
            slice: list of LaserPulseSlices each with an SRW wavefront object
            nslice: number of slices
            photon_e_ev: Photon energy [eV]
            sig_s: RMS bunch length [m]
            _lambda0: central wavelength [m]
            _sxvals: RMS horizontal beam size of each slice [m]
            _syvals: RMS vertical beam size of each slice [m]
    """

    _INPUT_ERROR = InvalidLaserPulseInputError
    _DEFAULTS = _LASER_PULSE_DEFAULTS

    def __init__(self, params=None, files=None):
        params = self._get_params(params)

        self._validate_params(params, files)
        # instantiate the array of slices
        self.slice = []
        self.files = files
        self.pulse_direction = params.pulse_direction
        self.flatten_cutoff = params.phase_flatten_cutoff
        self.sigx_waist = params.sigx_waist
        self.sigy_waist = params.sigy_waist
        self.num_sig_trans = params.num_sig_trans
        self.nslice = params.nslice
        self.photon_e_ev = params.photon_e_ev
        self.sig_s = params.tau_fwhm * const.c / 2.355
        self.num_sig_long = params.num_sig_long
        self._lambda0 = abs(
            units.calculate_lambda0_from_phE(params.photon_e_ev * const.e)
        )  # Function requires energy in J
        self.pulseE = params.pulseE

        assert (
            params.tau_fwhm > 10.0 * params.tau_0
        ), "ERROR -- Invalid pulse length parameters provided"

        # A factor to match experimental red-shift, compensating for
        # pulse stretcher implementation
        params.tau_0 *= 2.0

        tau_fwhm_intensity_profile = params.tau_fwhm / np.sqrt(2.0)
        tau_0_intensity_profile = params.tau_0 / np.sqrt(2.0)

        if tau_0_intensity_profile == 0.0:
            self.initial_chirp = 0.0

            d_nu_fwhm = (2.0 * np.log(2.0)) / (np.pi * tau_fwhm_intensity_profile)

        else:
            alpha = 2.0 * np.log(2.0)  # natural log
            self.initial_chirp = alpha / (
                tau_fwhm_intensity_profile * tau_0_intensity_profile
            )

            d_nu_fwhm = (2.0 * np.log(2.0)) / (np.pi * tau_0_intensity_profile)

        nu_fraction = d_nu_fwhm / (const.c / self._lambda0)
        self.d_lambda_RMS = (
            nu_fraction * self._lambda0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        params.d_lambda_RMS = self.d_lambda_RMS

        for i in range(params.nslice):
            # add the slices; each (slowly) instantiates an SRW wavefront object
            self.slice.append(LaserPulseSlice(i, params.copy(), files=self.files))
        self._sxvals = []  # horizontal slice data
        self._syvals = []  # vertical slice data

    def resize_laser_mesh(self):
        # Manually force mesh back to original extent + number of cells

        def _resize(initial_laser_xy, photon_e_ev, wfr0):
            new_x = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            new_y = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)

            # Check if need to make changes
            if np.array_equal(new_x, initial_laser_xy.x) and np.array_equal(
                new_y, initial_laser_xy.y
            ):
                return wfr0

            else:
                re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d = srwutil.extract_2d_fields(wfr0)

                # Interpolate ex meshes
                rect_biv_spline_xre = RectBivariateSpline(new_x, new_y, re_ex_2d)
                rect_biv_spline_xim = RectBivariateSpline(new_x, new_y, im_ex_2d)
                new_re_ex_2d = rect_biv_spline_xre(
                    initial_laser_xy.x, initial_laser_xy.y
                )
                new_im_ex_2d = rect_biv_spline_xim(
                    initial_laser_xy.x, initial_laser_xy.y
                )

                # Interpolate ey meshes
                rect_biv_spline_yre = RectBivariateSpline(new_x, new_y, re_ey_2d)
                rect_biv_spline_yim = RectBivariateSpline(new_x, new_y, im_ey_2d)
                new_re_ey_2d = rect_biv_spline_yre(
                    initial_laser_xy.x, initial_laser_xy.y
                )
                new_im_ey_2d = rect_biv_spline_yim(
                    initial_laser_xy.x, initial_laser_xy.y
                )

                wfr = srwutil.make_wavefront(
                    new_re_ex_2d,
                    new_im_ex_2d,
                    new_re_ey_2d,
                    new_im_ey_2d,
                    photon_e_ev,
                    initial_laser_xy.x,
                    initial_laser_xy.y,
                )

                return wfr

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            thisSlice.wfr = _resize(
                thisSlice.initial_laser_xy, thisSlice.photon_e_ev, thisSlice.wfr
            )

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.wfr = _resize(
                    thisSubSlice.initial_laser_xy,
                    thisSubSlice.photon_e_ev,
                    thisSubSlice.wfr,
                )

    def flatten_phase_edges(self):
        def _flatten_edges(flatten_cutoff, photon_e_ev, wfr0):
            # identify radial distance of every cell
            x = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            y = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)
            xv, yv = np.meshgrid(x, y)
            r = np.sqrt(xv**2.0 + yv**2.0)

            # extract the intensity and phase for all laser pulses
            intensity_2d = srwutil.calc_int_from_elec(wfr0)
            phase_1d = srwlib.array("d", [0] * wfr0.mesh.nx * wfr0.mesh.ny)
            srwl.CalcIntFromElecField(phase_1d, wfr0, 0, 4, 3, wfr0.mesh.eStart, 0, 0)
            phase_2d = unwrap_phase(
                np.array(phase_1d)
                .reshape((wfr0.mesh.nx, wfr0.mesh.ny), order="C")
                .astype(np.float64)
            )

            location_flatten = np.where(np.abs(r) >= (flatten_cutoff * wfr0.mesh.xFin))

            x_average = x[(np.abs(x - (flatten_cutoff * wfr0.mesh.xFin))).argmin()]
            location_value = np.where(np.abs(r - x_average) <= np.diff(x)[0] / 2.0)
            flatten_value = np.mean(phase_2d[location_value])

            phase_2d[location_flatten] = flatten_value

            e_norm = np.sqrt(2.0 * intensity_2d / (const.c * const.epsilon_0))
            re_ex = np.multiply(e_norm, np.cos(phase_2d))
            im_ex = np.multiply(e_norm, np.sin(phase_2d))
            re_ey = np.zeros(np.shape(re_ex))
            im_ey = np.zeros(np.shape(im_ex))

            # remake the wavefront
            wfr = srwutil.make_wavefront(re_ex, im_ex, re_ey, im_ey, photon_e_ev, x, y)
            return wfr

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            thisSlice.wfr = _flatten_edges(
                self.flatten_cutoff, thisSlice.photon_e_ev, thisSlice.wfr
            )

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.wfr = _flatten_edges(
                    self.flatten_cutoff, thisSubSlice.photon_e_ev, thisSubSlice.wfr
                )

    def extract_total_2d_phase(self):

        bw_nslice = self.slice[0].bw_nslice
        photon_number = np.zeros((bw_nslice + 1, self.nslice))
        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            photon_number[0, j] = np.sum(thisSlice.n_photons_2d.mesh)

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                photon_number[k + 1, j] = np.sum(thisSubSlice.n_photons_2d.mesh)

        def _extract_phase(j, k, photon_number, wfr0):
            slice_phase_1d = srwlib.array("d", [0] * wfr0.mesh.nx * wfr0.mesh.ny)
            srwlib.srwl.CalcIntFromElecField(
                slice_phase_1d, wfr0, 0, 4, 3, wfr0.mesh.eStart, 0, 0
            )
            slice_phase_2d = unwrap_phase(
                np.array(slice_phase_1d)
                .reshape((wfr0.mesh.nx, wfr0.mesh.ny), order="C")
                .astype(np.float64)
            )

            weight = photon_number[k, j] / np.sum(photon_number)
            return slice_phase_2d * weight

        total_phase_2d = np.zeros(
            (self.slice[0].wfr.mesh.nx, self.slice[0].wfr.mesh.ny)
        )

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            total_phase_2d += _extract_phase(j, 0, photon_number, thisSlice.wfr)

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                total_phase_2d += _extract_phase(
                    j, k + 1, photon_number, thisSubSlice.wfr
                )

        return total_phase_2d

    def extract_total_2d_elec_fields(self):
        # Assumes gaussian shape

        nslices_pulse = len(self.slice)

        # Assumes each slice has the same dimensions
        e_total = PKDict(
            re=np.zeros(
                (self.slice[0].wfr.mesh.nx, self.slice[0].wfr.mesh.ny, nslices_pulse)
            ),
            im=np.zeros(
                (self.slice[0].wfr.mesh.nx, self.slice[0].wfr.mesh.ny, nslices_pulse)
            ),
        )

        pulse_end1 = (self.slice[0]._pulse_pos - 0.5 * self.slice[0].ds) / (
            2.0 * self.slice[0].sig_s
        )
        pulse_end2 = (self.slice[-1]._pulse_pos + 0.5 * self.slice[-1].ds) / (
            2.0 * self.slice[-1].sig_s
        )
        pulse_factor = special.erf(pulse_end2) - special.erf(pulse_end1)

        def _extract_elec_fields(pulse_factor, pulse_pos, ds, sig_s, wfr0):

            # total component of electric field
            re0, re0_mesh = srwutil.calc_int_from_wfr(
                wfr0, _pol=6, _int_type=5, _det=None, _fname="", _pr=False
            )
            im0, im0_mesh = srwutil.calc_int_from_wfr(
                wfr0, _pol=6, _int_type=6, _det=None, _fname="", _pr=False
            )

            e_total_re = (
                np.array(re0)
                .reshape((wfr0.mesh.nx, wfr0.mesh.ny), order="C")
                .astype(np.float64)
            )
            e_total_im = (
                np.array(im0)
                .reshape((wfr0.mesh.nx, wfr0.mesh.ny), order="C")
                .astype(np.float64)
            )

            # gaussian scale
            slice_end1 = (pulse_pos - 0.5 * ds) / (2.0 * sig_s)
            slice_end2 = (pulse_pos + 0.5 * ds) / (2.0 * sig_s)
            slice_width_factor = (
                1.0 / np.exp(-(pulse_pos**2.0) / (2.0 * sig_s) ** 2.0)
            ) * ((special.erf(slice_end2) - special.erf(slice_end1)) / pulse_factor)

            # scale slice fields by factor to represent full slice, not just middle value
            e_total_re *= slice_width_factor
            e_total_im *= slice_width_factor

            return e_total_re, e_total_im

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            pulse_pos = thisSlice._pulse_pos
            ds = thisSlice.ds
            sig_s = thisSlice.sig_s
            e_total.re[:, :, j], e_total.im[:, :, j] = _extract_elec_fields(
                pulse_factor, pulse_pos, ds, sig_s, thisSlice.wfr
            )
            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                e_t_re, e_t_im = _extract_elec_fields(
                    pulse_factor, pulse_pos, ds, sig_s, thisSubSlice.wfr
                )
                e_total.re[:, :, j] += e_t_re
                e_total.im[:, :, j] += e_t_im

        e_total.re = np.sum(e_total.re, axis=2)
        e_total.im = np.sum(e_total.im, axis=2)

        return e_total

    def shift_wavefront(
        self,
        pump_offset_x,
        pump_offset_y,
    ):
        def _shift_wfr(pump_offset_x, pump_offset_y, photon_e_ev, wfr0):
            x = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            y = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)

            new_x = x - pump_offset_x
            new_y = y - pump_offset_y

            re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d = srwutil.extract_2d_fields(wfr0)

            wfr = srwutil.make_wavefront(
                re_ex_2d,
                im_ex_2d,
                re_ey_2d,
                im_ey_2d,
                photon_e_ev,
                new_x,
                new_y,
            )
            return wfr

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            thisSlice.wfr = _shift_wfr(
                pump_offset_x, pump_offset_y, thisSlice.photon_e_ev, thisSlice.wfr
            )

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.wfr = _shift_wfr(
                    pump_offset_x,
                    pump_offset_y,
                    thisSubSlice.photon_e_ev,
                    thisSubSlice.wfr,
                )

    def combine_n2_variation(
        self,
        laser_pulse_copies,
        radial_n2_factor,
        pump_waist,
        max_n2,
    ):
        def _combine_variation(
            radial_n2_factor, pump_waist, max_n2, photon_e_ev, wfr_0, wfr_max
        ):

            # extract the intensity and phase for all laser pulses
            intensity_2d = PKDict(
                n2_max=srwutil.calc_int_from_elec(wfr_max),
                n2_0=srwutil.calc_int_from_elec(wfr_0),
            )

            phase_1d_max = srwlib.array("d", [0] * wfr_max.mesh.nx * wfr_max.mesh.ny)
            phase_1d_0 = srwlib.array("d", [0] * wfr_0.mesh.nx * wfr_0.mesh.ny)
            srwl.CalcIntFromElecField(
                phase_1d_max, wfr_max, 0, 4, 3, wfr_max.mesh.eStart, 0, 0
            )
            srwl.CalcIntFromElecField(
                phase_1d_0, wfr_0, 0, 4, 3, wfr_0.mesh.eStart, 0, 0
            )
            phase_2d = PKDict(
                n2_max=unwrap_phase(
                    np.array(phase_1d_max)
                    .reshape((wfr_max.mesh.nx, wfr_max.mesh.ny), order="C")
                    .astype(np.float64)
                ),
                n2_0=unwrap_phase(
                    np.array(phase_1d_0)
                    .reshape((wfr_0.mesh.nx, wfr_0.mesh.ny), order="C")
                    .astype(np.float64)
                ),
            )

            # identify radial distance of every cell
            x = np.linspace(wfr_0.mesh.xStart, wfr_0.mesh.xFin, wfr_0.mesh.nx)
            y = np.linspace(wfr_0.mesh.yStart, wfr_0.mesh.yFin, wfr_0.mesh.ny)
            xv, yv = np.meshgrid(x, y)
            r = np.sqrt(xv**2.0 + yv**2.0)

            # Calculate the phase shift value
            zero_cut_off = (
                radial_n2_factor * pump_waist
            )  # Outside this value is zero n2
            location_max = np.where(np.abs(r - zero_cut_off) <= np.diff(x)[0] / 2.0)
            location_0 = np.where(
                np.abs(r - (zero_cut_off + np.diff(x)[0])) <= np.diff(x)[0] / 2.0
            )

            n2_max_average = np.mean(phase_2d.n2_max[location_max])
            n2_0_average = np.mean(phase_2d.n2_0[location_0])
            shift_value = n2_max_average - n2_0_average
            phase_2d.n2_0 += shift_value

            # Assign the n2 = 0 fields initially
            intensity = intensity_2d.n2_0
            phase = phase_2d.n2_0
            n2 = np.zeros(np.shape(intensity))

            # Calculate the scaling function
            scaling_fn = np.zeros(np.shape(r))
            location = np.where(np.abs(r) <= zero_cut_off + (np.diff(x)[0] / 2.0))

            xv_temp = (xv / (zero_cut_off + (np.diff(x)[0] / 2.0))) * np.pi
            yv_temp = (yv / (zero_cut_off + (np.diff(x)[0] / 2.0))) * np.pi
            scaling_fn[location] = (
                np.cos(np.sqrt(xv_temp[location] ** 2.0 + yv_temp[location] ** 2.0)) + 1
            ) / 2.0

            n2[location] = ((1.0 - scaling_fn[location]) * 0.0) + (
                scaling_fn[location] * max_n2
            )
            intensity[location] = (
                (1.0 - scaling_fn[location]) * intensity_2d.n2_0[location]
            ) + (scaling_fn[location] * intensity_2d.n2_max[location])
            phase[location] = (
                (1.0 - scaling_fn[location]) * phase_2d.n2_0[location]
            ) + (scaling_fn[location] * phase_2d.n2_max[location])

            # Calculate the fields
            e_norm = np.sqrt(2.0 * intensity / (const.c * const.epsilon_0))
            re_ex = np.multiply(e_norm, np.cos(phase))
            im_ex = np.multiply(e_norm, np.sin(phase))
            re_ey = np.zeros(np.shape(re_ex))
            im_ey = np.zeros(np.shape(im_ex))

            # remake the wavefront
            wfr = srwutil.make_wavefront(re_ex, im_ex, re_ey, im_ey, photon_e_ev, x, y)
            return wfr

        for j in np.arange(self.nslice):
            # For each laser slice, combine the propagated wavefronts

            thisSlice = self.slice[j]
            thisSlice.wfr = _combine_variation(
                radial_n2_factor,
                pump_waist,
                max_n2,
                thisSlice.photon_e_ev,
                laser_pulse_copies.n2_0.slice[j].wfr,
                laser_pulse_copies.n2_max.slice[j].wfr,
            )
            thisSlice.n_photons_2d.mesh = laser_pulse_copies.n2_0.slice[
                j
            ].n_photons_2d.mesh

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.wfr = _combine_variation(
                    radial_n2_factor,
                    pump_waist,
                    max_n2,
                    thisSubSlice.photon_e_ev,
                    laser_pulse_copies.n2_0.slice[j].bandwidth_slice[k].wfr,
                    laser_pulse_copies.n2_max.slice[j].bandwidth_slice[k].wfr,
                )
                thisSubSlice.n_photons_2d.mesh = (
                    laser_pulse_copies.n2_0.slice[j]
                    .bandwidth_slice[k]
                    .n_photons_2d.mesh
                )

        return self

    def ideal_mirror_180(self):
        # 0.0 corresponds to 'right' or 'x=0,y=0,z' or 'forward'
        if self.pulse_direction == 0.0:
            self.pulse_direction = 180.0
        elif self.pulse_direction == 180.0:
            self.pulse_direction = 0.0

        def _flip_fields(photon_e_ev, initial_laser_xy, wfr0):
            re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d = srwutil.extract_2d_fields(wfr0)

            # E_f (x,y) = E_i (-x,-y)
            new_re_ex_2d = np.flip(re_ex_2d)
            new_im_ex_2d = np.flip(im_ex_2d)
            new_re_ey_2d = np.flip(re_ey_2d)
            new_im_ey_2d = np.flip(im_ey_2d)

            wfr = srwutil.make_wavefront(
                new_re_ex_2d,
                new_im_ex_2d,
                new_re_ey_2d,
                new_im_ey_2d,
                photon_e_ev,
                initial_laser_xy.x,
                initial_laser_xy.y,
            )

            return wfr

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            thisSlice.wfr = _flip_fields(
                thisSlice.photon_e_ev, thisSlice.initial_laser_xy, thisSlice.wfr
            )

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.wfr = _flip_fields(
                    thisSubSlice.photon_e_ev,
                    thisSubSlice.initial_laser_xy,
                    thisSubSlice.wfr,
                )

    def zero_phase(self):
        # Manually zero the phase
        def _zero_wfr_phase(photon_e_ev, initial_laser_xy, wfr0):

            re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d = srwutil.extract_2d_fields(wfr0)

            new_re_ex_2d = np.sqrt(re_ex_2d**2.0 + im_ex_2d**2.0)
            new_im_ex_2d = np.zeros(np.shape(new_re_ex_2d))
            new_re_ey_2d = np.zeros(np.shape(new_re_ex_2d))
            new_im_ey_2d = np.zeros(np.shape(new_re_ex_2d))

            # remake the wavefront
            wfr = srwutil.make_wavefront(
                new_re_ex_2d,
                new_im_ex_2d,
                new_re_ey_2d,
                new_im_ey_2d,
                photon_e_ev,
                initial_laser_xy.x,
                initial_laser_xy.y,
            )
            return wfr

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            thisSlice.wfr = _zero_wfr_phase(
                thisSlice.photon_e_ev, thisSlice.initial_laser_xy, thisSlice.wfr
            )

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.wfr = _zero_wfr_phase(
                    thisSubSlice.photon_e_ev,
                    thisSubSlice.initial_laser_xy,
                    thisSubSlice.wfr,
                )

    def update_photon_positions(self):
        def _update_ph_pos(pulse_pos, ds, sig_s, photon_e, wfr, init_n_photons):

            intens_2d = srwutil.calc_int_from_elec(wfr)
            efield_abs_sqrd_2d = (
                np.sqrt(const.mu_0 / const.epsilon_0) * 2.0 * intens_2d
            )  # [V^2/m^2]

            dx = (wfr.mesh.xFin - wfr.mesh.xStart) / wfr.mesh.nx
            dy = (wfr.mesh.yFin - wfr.mesh.yStart) / wfr.mesh.ny
            cell_area = dx * dy

            # Field energy per grid cell is the area of that cell times the energy density
            end1 = (pulse_pos - 0.5 * ds) / (np.sqrt(2.0) * sig_s)
            end2 = (pulse_pos + 0.5 * ds) / (np.sqrt(2.0) * sig_s)
            energy_2d = (
                cell_area
                * (const.epsilon_0 / 2.0)
                * (
                    efield_abs_sqrd_2d
                    / np.exp(-(pulse_pos**2.0) / (np.sqrt(2.0) * sig_s) ** 2.0)
                )
                * (
                    (np.sqrt(np.pi) / 2.0)
                    * (np.sqrt(2.0) * sig_s)
                    * (special.erf(end2) - special.erf(end1))
                )
            )

            new_n_photons = copy.deepcopy(init_n_photons)

            # Number of photons in each grid cell can be found by dividing the
            # total energy of the laser in that grid cell by the energy of a photon
            new_n_photons.mesh = energy_2d / photon_e
            new_n_photons.x = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx)
            new_n_photons.y = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)

            return new_n_photons

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]

            pulse_pos = thisSlice._pulse_pos
            ds = thisSlice.ds
            sig_s = thisSlice.sig_s

            thisSlice.n_photons_2d = _update_ph_pos(
                pulse_pos,
                ds,
                sig_s,
                thisSlice.photon_e_ev * const.e,
                thisSlice.wfr,
                thisSlice.n_photons_2d,
            )

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                thisSubSlice.n_photons_2d = _update_ph_pos(
                    pulse_pos,
                    ds,
                    sig_s,
                    thisSubSlice.photon_e_ev * const.e,
                    thisSubSlice.wfr,
                    thisSubSlice.n_photons_2d,
                )

    def central_and_mean_wavelength(self, plot=False):

        wavelength_mesh = np.zeros((self.slice[0].bw_nslice + 1, self.nslice))
        photon_mesh = np.zeros((self.slice[0].bw_nslice + 1, self.nslice))

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]
            wavelength_mesh[3, j] = abs(
                units.calculate_lambda0_from_phE(thisSlice.photon_e_ev * const.e)
            )
            photon_mesh[3, j] = np.sum(thisSlice.n_photons_2d.mesh)
            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]
                bw_ind = int(3 + thisSubSlice.sign * thisSubSlice.rms_int)
                wavelength_mesh[bw_ind, j] = abs(
                    units.calculate_lambda0_from_phE(thisSubSlice.photon_e_ev * const.e)
                )
                photon_mesh[bw_ind, j] = np.sum(thisSubSlice.n_photons_2d.mesh)

        mean_wavelength = np.sum(wavelength_mesh * photon_mesh / np.sum(photon_mesh))

        wavelength, photon_n = bin_arrays(
            (photon_mesh).flatten(), (wavelength_mesh * 1e9).flatten(), self.nslice
        )
        cs = CubicSpline(wavelength, photon_n)
        wavelength_high_res = np.linspace(
            wavelength[0], wavelength[-1], len(wavelength) * 10
        )
        photon_n_high_res = cs(wavelength_high_res)

        loc = np.argwhere(np.max(photon_n_high_res) == photon_n_high_res)
        central_wavelength = wavelength_high_res[loc]

        if plot:
            plt.figure()
            plt.plot(wavelength, photon_n, "k", label="Pulse Data")
            plt.plot(
                wavelength_high_res, photon_n_high_res, "--r", label="Cubic Spline Fit"
            )
            plt.plot(
                wavelength_high_res[loc],
                photon_n_high_res[loc],
                "*",
                label="Central Wavelength",
            )
            plt.legend()
            plt.show()

        return central_wavelength, mean_wavelength

    def calc_total_energy(self):
        bw_nslice = self.slice[0].bw_nslice
        photon_e_ev = np.zeros((bw_nslice + 1, self.nslice))
        photon_number = np.zeros((bw_nslice + 1, self.nslice))

        for j in np.arange(self.nslice):
            thisSlice = self.slice[j]

            photon_e_ev[0, j] = thisSlice.photon_e_ev
            photon_number[0, j] = np.sum(thisSlice.n_photons_2d.mesh)

            for k in np.arange(thisSlice.bw_nslice):
                thisSubSlice = thisSlice.bandwidth_slice[k]

                photon_e_ev[k + 1, j] = thisSubSlice.photon_e_ev
                photon_number[k + 1, j] = np.sum(thisSubSlice.n_photons_2d.mesh)

        pulse_energy = photon_number * photon_e_ev * const.e
        return np.sum(pulse_energy)

    def _validate_params(self, input_params, files):
        # if files and input_params.nslice > 1:
        #     raise self._INPUT_ERROR("cannot use file inputs with more than one slice")
        super()._validate_params(input_params)

    def compute_middle_slice_intensity(self):
        wfr = self.slice[len(self.slice) // 2].wfr
        (ar2d, sx, sy, xavg, yavg) = rswf.rmsWavefrontIntensity(wfr)
        self._sxvals.append(sx)
        self._syvals.append(sy)
        return (wfr, ar2d, sx, sy, xavg, yavg)

    def rmsvals(self):
        sx = []
        sy = []
        for sl in self.slice:
            (_, sigx, sigy, _, _) = rswf.rmsWavefrontIntensity(sl.wfr)
            sx.append(sigx)
            sy.append(sigy)

        return (sx, sy)

    def intensity_vals(self):
        return [rswf.maxWavefrontIntensity(s.wfr) for s in self.slice]

    def pulsePos(self):
        return [s._pulse_pos for s in self.slice]

    def energyvals(self):
        return [s.photon_e_ev for s in self.slice]

    def slice_wfr(self, slice_index):
        return self.slice[slice_index].wfr


class LaserPulseSlice(ValidatorBase):
    """
    This class represents a longitudinal slice in a laser pulse.
    There will be a number of wavefronts each with different wavelengths (energy).
    The slice is composed of an SRW wavefront object, which is defined here:
    https://github.com/ochubar/SRW/blob/master/env/work/srw_python/srwlib.py#L2048

    Assumes a longitudinal gaussian profile when initializing

    Args:
        slice_index (int): index of slice
        params (PKDict): accepts input params from LaserPulse class __init__

    Returns:
        instance of class
    """

    _INPUT_ERROR = InvalidLaserPulseInputError
    _DEFAULTS = _LASER_PULSE_DEFAULTS

    def __init__(self, slice_index, params=None, files=None):
        self._validate_type(slice_index, int, "slice_index")
        params = self._get_params(params)
        self._validate_params(params)
        self._lambda0 = units.calculate_lambda0_from_phE(
            params.photon_e_ev * const.e
        )  # Function requires energy in J
        self.slice_index = slice_index
        self.sigx_waist = params.sigx_waist
        self.sigy_waist = params.sigy_waist
        self.num_sig_trans = params.num_sig_trans
        self.nslice = params.nslice
        self.nx_slice = params.nx_slice
        self.dist_waist = params.dist_waist
        self.d_lambda_RMS = params.d_lambda_RMS

        self.pulseE_slice = (
            params.pulseE / self.nslice
        )  # currently assumes consistent length and energy across all slices

        self.sig_s = params.tau_fwhm * const.c / 2.355
        self.num_sig_long = params.num_sig_long
        self.ds = (
            2 * params.num_sig_long * self.sig_s / params.nslice
        )  # longitudinal spacing between slices
        self._pulse_pos = (
            -params.num_sig_long * self.sig_s + (slice_index + 0.5) * self.ds
        )

        tau_fwhm_intensity_profile = params.tau_fwhm / np.sqrt(2.0)
        tau_0_intensity_profile = params.tau_0 / np.sqrt(2.0)

        if tau_0_intensity_profile == 0.0:
            chirp = 0.0
        else:
            alpha = 2.0 * np.log(2.0)  # natural log
            chirp = alpha / (tau_fwhm_intensity_profile * tau_0_intensity_profile)

        nu = (const.c / self._lambda0) + (chirp / np.pi) * (self._pulse_pos / const.c)
        self._lambda = const.c / nu
        self.photon_e_ev = units.calculate_phE_from_lambda0(self._lambda) / const.e

        self._wavefront(params, files)

        # Calculate the initial number of photons in each slice from pulseE_slice
        self.n_photons_2d = self._calc_init_n_photons()

        self.initial_laser_xy = PKDict(
            x=np.linspace(self.wfr.mesh.xStart, self.wfr.mesh.xFin, self.wfr.mesh.nx),
            y=np.linspace(self.wfr.mesh.yStart, self.wfr.mesh.yFin, self.wfr.mesh.ny),
        )

        self.bandwidth_slice = []
        self.bw_nslice = 6
        for i in range(self.bw_nslice):
            self.bandwidth_slice.append(
                BandwidthSlice(
                    i,
                    np.copy(self.d_lambda_RMS),
                    np.copy(self._lambda),
                    copy.deepcopy(self.wfr),
                    copy.deepcopy(self.initial_laser_xy),
                    np.copy(self.pulseE_slice),
                    copy.deepcopy(self.n_photons_2d),
                )
            )

        # fraction of photons contained in the sub-slice corresponding to the central wavelength of that laser pulse slice
        # is an integral from integral_start to integral_end
        sigma = self.d_lambda_RMS * np.sqrt(2.0)
        erf_arg1 = (-0.5 * self.d_lambda_RMS) / (np.sqrt(2.0) * sigma)
        erf_arg2 = (0.5 * self.d_lambda_RMS) / (np.sqrt(2.0) * sigma)
        subslice_fraction = 0.5 * (special.erf(erf_arg2) - special.erf(erf_arg1))

        self.pulseE_slice *= subslice_fraction
        self.n_photons_2d.mesh *= subslice_fraction

        re0_2d_ex, im0_2d_ex, re0_2d_ey, im0_2d_ey = srwutil.extract_2d_fields(self.wfr)

        re0_2d_ex *= np.sqrt(subslice_fraction)
        im0_2d_ex *= np.sqrt(subslice_fraction)
        re0_2d_ey *= np.sqrt(subslice_fraction)
        im0_2d_ey *= np.sqrt(subslice_fraction)

        self.wfr = srwutil.make_wavefront(
            re0_2d_ex,
            im0_2d_ex,
            re0_2d_ey,
            im0_2d_ey,
            self.photon_e_ev,
            self.initial_laser_xy.x,
            self.initial_laser_xy.y,
        )

    def _wavefront(self, params, files):
        if files:
            with open(files.meta) as fh:
                for line in fh:
                    if line.startswith("pixel_size_h_microns"):
                        pixel_size_h = float(
                            line.split(":")[-1].split(",")[0]
                        )  # microns
                    if line.startswith("pixel_size_v_microns"):
                        pixel_size_v = float(
                            line.split(":")[-1].split(",")[0]
                        )  # microns

            # central wavelength of the laser pulse
            lambda0_micron = self._lambda0 * (1.0e6)  # 0.8

            # parse the ccd_data and wfs_data (measured phases of the wavefront)
            ccd_data = np.genfromtxt(files.ccd, skip_header=1)
            wfs_data = np.genfromtxt(files.wfs, skip_header=1, skip_footer=0)

            # clean up any NaN's
            ccd_data[np.isnan(ccd_data)] = 0.0

            if np.sum(np.isnan(wfs_data)) != 0:
                wfs_data = _replace_phase_nan(wfs_data)

            nx_wfs = np.shape(wfs_data)[0]
            ny_wfs = np.shape(wfs_data)[1]
            nx_ccd = np.shape(ccd_data)[0]
            ny_ccd = np.shape(ccd_data)[1]

            edges = np.concatenate(
                (wfs_data[0, :], wfs_data[1:, -1], wfs_data[-1, :-1], wfs_data[1:-1, 0])
            )
            edge_average = np.mean(edges)

            # Increase the shape to 64x64: pad wfs data with array edge, pad ccd data with zeros
            if nx_wfs < 64:
                wfs_data = np.pad(
                    wfs_data,
                    ((int((64 - nx_wfs) / 2), int((64 - nx_wfs) / 2)), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                )
            if ny_wfs < 64:
                wfs_data = np.pad(
                    wfs_data,
                    ((0, 0), (int((64 - ny_wfs) / 2), int((64 - ny_wfs) / 2))),
                    mode="constant",
                    constant_values=np.nan,
                )
            if nx_ccd < 64:
                ccd_data = np.pad(
                    ccd_data,
                    ((int((64 - nx_ccd) / 2), int((64 - nx_ccd) / 2)), (0, 0)),
                    mode="constant",
                )
            if ny_ccd < 64:
                ccd_data = np.pad(
                    ccd_data,
                    ((0, 0), (int((64 - ny_ccd) / 2), int((64 - ny_ccd) / 2))),
                    mode="constant",
                )

            wfs_data = _replace_phase_nan(wfs_data)
            ccd_data = gaussian_pad(ccd_data)

            assert np.shape(wfs_data) == np.shape(
                ccd_data
            ), "ERROR -- WFS and CCD data have diferent shapes!!"

            nx = np.shape(wfs_data)[0]
            ny = np.shape(wfs_data)[1]
            assert (
                nx == ny
            ), "ERROR -- data is not square"  # Add method to square data if it is larger than 64x64?

            # convert from microns to radians
            rad_per_micron = math.pi / lambda0_micron
            wfs_data *= rad_per_micron

            # create the x,y arrays with physical units based on the diagnostic pixel dimensions
            x_max = 0.5 * (nx + 1.0) * pixel_size_h * 1.0e-6  # [m]
            x_min = -x_max
            y_max = 0.5 * (ny + 1.0) * pixel_size_v * 1.0e-6  # [m]
            y_min = -y_max

            # Calulate the real and imaginary parts of the Ex,Ey electric field components
            e_norm = np.sqrt(ccd_data)

            ex_real = np.multiply(e_norm, np.cos(wfs_data))
            ex_imag = np.multiply(e_norm, np.sin(wfs_data))

            # scale the wfr sensor data
            self.wfs_norm_factor = 3378.048302768955  # 2841.7370456965646

            # scale for slice location
            number_slices_correction = np.exp(
                -self._pulse_pos**2.0 / (2.0 * self.sig_s**2.0)
            )

            ex_real *= number_slices_correction * self.wfs_norm_factor
            ex_imag *= number_slices_correction * self.wfs_norm_factor

            x = np.linspace(x_min, x_max, nx)
            y = np.linspace(y_min, y_max, ny)
            self.wfr = srwutil.make_wavefront(
                ex_real,
                ex_imag,
                np.zeros(np.shape(ex_real)),
                np.zeros(np.shape(ex_imag)),
                self.photon_e_ev,
                x,
                y,
            )

            return

        # Since pulseE = fwhm_tau * spot_size * intensity, new_pulseE = old_pulseE / fwhm_tau
        # Adjust for the length of the pulse + a constant factor to make pulseE = sum(energy_2d)
        constant_factor = 1.198945869831954  # 7.3948753166511745
        length_factor = constant_factor / (self.sig_s / params.nslice)  # / self.ds

        # calculate field energy in this slice
        sliceEnInt = (
            length_factor
            * self.pulseE_slice
            * np.exp(-self._pulse_pos**2 / (2 * self.sig_s**2))
        )

        self.wfr = srwutil.createGsnSrcSRW(
            self.sigx_waist,
            self.sigy_waist,
            self.num_sig_trans,
            self._pulse_pos,
            sliceEnInt,
            params.poltype,
            self.nx_slice,
            self.nx_slice,
            self.photon_e_ev,
            params.mx,
            params.my,
        )

    def _calc_init_n_photons(self):

        # Note: assumes longitudinal gaussian profile when initializing

        intens_2d = srwutil.calc_int_from_elec(self.wfr)  # extract 2d intensity

        efield_abs_sqrd_2d = (
            np.sqrt(const.mu_0 / const.epsilon_0) * 2.0 * intens_2d
        )  # [V^2/m^2]

        dx = (self.wfr.mesh.xFin - self.wfr.mesh.xStart) / self.wfr.mesh.nx
        dy = (self.wfr.mesh.yFin - self.wfr.mesh.yStart) / self.wfr.mesh.ny

        # Field energy per grid cell is the area of that cell times the energy density
        cell_area = dx * dy
        end1 = (self._pulse_pos - 0.5 * self.ds) / (np.sqrt(2.0) * self.sig_s)
        end2 = (self._pulse_pos + 0.5 * self.ds) / (np.sqrt(2.0) * self.sig_s)
        energy_2d = (
            cell_area
            * (const.epsilon_0 / 2.0)
            * (
                efield_abs_sqrd_2d
                / np.exp(-self._pulse_pos**2.0 / (np.sqrt(2.0) * self.sig_s) ** 2.0)
            )
            * (
                (np.sqrt(np.pi) / 2.0)
                * (np.sqrt(2.0) * self.sig_s)
                * (special.erf(end2) - special.erf(end1))
            )
        )

        # Get slice value of photon_e (will be in eV)
        photon_e = self.photon_e_ev * const.e

        # Number of photons in each grid cell can be found by dividing the
        # total energy of the laser in that grid cell by the energy of a photon
        n_photons_2d = PKDict(
            mesh=(energy_2d / photon_e),
            x=np.linspace(self.wfr.mesh.xStart, self.wfr.mesh.xFin, self.wfr.mesh.nx),
            y=np.linspace(self.wfr.mesh.yStart, self.wfr.mesh.yFin, self.wfr.mesh.ny),
        )
        return n_photons_2d


class BandwidthSlice(ValidatorBase):
    """
    This class represents the bandwidth of each slice in a laser pulse.
    Each of our existing laser pulse slices are divided into 7 overlapping
    slices, (the parent/original + 6 additional) with central wavelength values of
        lambda_c
        lambda_c +/- 1 * d_lambda_RMS
        lambda_c +/- 2 * d_lambda_RMS
        lambda_c +/- 3 * d_lambda_RMS
    where lambda_c is the local value of the chirped wavelength for the parent.

    Args:
        slice_index (int): index of slice
        LaserSliceCopy: a copy of the slice these bandwidth slices beling to

    Returns:
        instance of class
    """

    _INPUT_ERROR = InvalidLaserPulseInputError
    _DEFAULTS = _LASER_PULSE_DEFAULTS

    def __init__(
        self,
        slice_index,
        d_lambda_RMS,
        slice_lambda,
        wfr,
        initial_laser_xy,
        original_pulseE_slice,
        original_n_photons_2d,
    ):

        self._validate_type(slice_index, int, "slice_index")
        self.slice_index = slice_index

        self.wfr = wfr
        self.initial_laser_xy = initial_laser_xy
        self.pulseE_slice = original_pulseE_slice
        self.n_photons_2d = original_n_photons_2d

        central_subslice_lambda = slice_lambda
        self.rms_int = np.ceil(2.0 * (self.slice_index + 1.0e-3) / 3.5)
        self.sign = 2.0 * (-self.slice_index % -2) + 1

        self._lambda = central_subslice_lambda + self.sign * self.rms_int * d_lambda_RMS
        self.photon_e_ev = units.calculate_phE_from_lambda0(self._lambda) / const.e

        # fraction of photons contained in the sub-slice corresponding to the central wavelength of that laser pulse slice
        # is an integral from integral_start to integral_end
        sigma = d_lambda_RMS * np.sqrt(2.0)
        if self.rms_int == 3:
            erf_arg1 = -1.0 * np.inf
            erf_arg2 = ((-1.0 * self.rms_int + 0.5) * d_lambda_RMS) / (
                np.sqrt(2.0) * sigma
            )
        else:
            erf_arg1 = ((self.sign * self.rms_int - 0.5) * d_lambda_RMS) / (
                np.sqrt(2.0) * sigma
            )
            erf_arg2 = ((self.sign * self.rms_int + 0.5) * d_lambda_RMS) / (
                np.sqrt(2.0) * sigma
            )
        subslice_fraction = 0.5 * (special.erf(erf_arg2) - special.erf(erf_arg1))

        self.pulseE_slice *= subslice_fraction
        self.n_photons_2d.mesh *= subslice_fraction

        re0_2d_ex, im0_2d_ex, re0_2d_ey, im0_2d_ey = srwutil.extract_2d_fields(self.wfr)

        re0_2d_ex *= np.sqrt(subslice_fraction)
        im0_2d_ex *= np.sqrt(subslice_fraction)
        re0_2d_ey *= np.sqrt(subslice_fraction)
        im0_2d_ey *= np.sqrt(subslice_fraction)

        self.wfr = srwutil.make_wavefront(
            re0_2d_ex,
            im0_2d_ex,
            re0_2d_ey,
            im0_2d_ey,
            self.photon_e_ev,
            self.initial_laser_xy.x,
            self.initial_laser_xy.y,
        )


class LaserPulseEnvelope(ValidatorBase):
    """Module defining a Hermite-Gaussian laser field of order (m,n).

    For now, we assume linear polarization of E along, x.
    Also, the evaluation is done for z=0 (for simplicity)
    The time variable t is ignored.

        Args:
            params (PKDict):
                required fields:
                    photon_e_ev (float): Photon energy [eV]
                    w0 (float): beamsize of laser pulse at waist [m]
                    a0 (float): laser amplitude, a=0.85e-9 lambda[micron] sqrt(I[W/cm^2])
                    dw0x (float): horizontal variation in waist size [m]
                    dw0y (float): vertical variation in waist size [m]
                    z_waist (float): longitudinal location of the waist [m]
                    dzwx (float): location (in z) of horizontal waist [m]
                    dzwy (float): location (in z) of vertical waist [m]
                    tau_fwhm (float): FWHM laser pulse length [s]
                    z_center (float): # longitudinal location of pulse center [m]
                    x_shift (float): horizontal shift of the spot center [m]
                    y_shift (float): vertical shift of the spot center [m]

    Returns:
        instance of class

    """

    _INPUT_ERROR = InvalidLaserPulseInputError
    _DEFAULTS = _ENVELOPE_DEFAULTS

    def __init__(self, params=None):
        params = self._get_params(params)
        self._validate_params(params)
        self.lambda0 = abs(
            units.calculate_lambda0_from_phE(params.photon_e_ev * const.e)
        )  # Function requires energy in J # central wavelength [m]
        # useful derived quantities
        self.k0 = rsc.TWO_PI / self.lambda0  # central wavenumber [radians/m]
        self.f0 = const.c / self.lambda0  # central frequency  [Hz]
        self.omega0 = rsc.TWO_PI * self.f0  # central angular frequency [radians/s]

        # Peak electric field [V/m]
        self.a0 = abs(params.a0)  # amplitude [dimensionless]
        # peak electric field [V/m]
        self.efield0 = self.a0 * const.m_e * self.omega0 * const.c / (const.e)

        # waist sizes and locations
        self.w0 = abs(params.w0)  # the waist size of the pulse
        self.set_waist_x(params.w0 + params.dw0x)  # horizontal waist size [m]
        self.set_waist_y(params.w0 + params.dw0y)  # vertical waist size [m]

        # self.z_waist = params.z_waist                 # the longitudinal location of the waist
        self.z_waist_x = params.dzwx  # location (in z) of horizontal waist
        self.z_waist_y = params.dzwy  # location (in z) of vertical waist

        # Rayleigh range
        self.zR = (
            0.5 * self.k0 * (self.w0) ** 2
        )  # Rayleigh range, ignoring horizontal/vertical differences
        #        print('\n ****** \n zR = ', self.zR)

        # pulse length
        self.tau_fwhm = params.tau_fwhm  # FWHM laser pulse length [s]
        self.L_fwhm = self.tau_fwhm * const.c  # FWHM laser pulse length [m]

        # longitudinal location of pulse center
        self.z_center = params.z_center

        # bulk transverse offsets of the laser pulse
        self.x_shift = params.x_shift  # horizontal shift of the spot center
        self.y_shift = params.y_shift  # vertical shift of the spot center

        # for now, we set the higher Hermite modes to zero
        self.setCoeffSingleModeX(0, 1.0)  # horizontal coefficients of Hermite expansion
        self.setCoeffSingleModeY(0, 1.0)  # vertical coefficients of Hermite expansion

        # for now, we set the rotation angle to zero
        self.wRotAngle = 0.0

        return

    def set_waist_x(self, _waistX):
        """
        set the horizontal waist size [m]

        Note:
            error handling; very small waist will cause performance issues
        """
        wFac = 4.0
        minSize = wFac * self.lambda0
        if _waistX >= minSize:
            self.waist_x = _waistX
        else:
            message = "waistX = " + str(_waistX) + "; must be >= " + str(minSize)
            raise Exception(message)

        # error handling; require that deviations from w0 are small
        self.dw0x = _waistX - self.w0
        if abs(self.dw0x) > 0.1 * self.w0:
            message = "dw0x/w0 = " + str(self.dw0x) + "; must be < 0.1 "
            raise Exception(message)

        self.piWxFac = math.sqrt(rsc.RT_2_OVER_PI / self.waist_x)
        self.zRx = 0.5 * self.k0 * self.waist_x**2  # horizintal Rayleigh range [m]
        self.qx0 = 0.0 + self.zRx * 1j
        return

    def set_waist_y(self, _waistY):
        """
        set the vertical waist size [m]

        Note:
            error handling; very small waist will cause performance issues
        """
        wFac = 4.0
        minSize = wFac * self.lambda0
        if _waistY >= minSize:
            self.waist_y = _waistY
        else:
            message = "waistY = " + str(_waistY) + "; must be >= " + str(minSize)
            raise Exception(message)

        # error handling; require that deviations from w0 are small
        self.dw0y = _waistY - self.w0
        if abs(self.dw0y) > 0.1 * self.w0:
            message = "dw0y/w0 = " + str(self.dw0y) + "; must be < 0.1 "
            raise Exception(message)

        self.piWyFac = math.sqrt(rsc.RT_2_OVER_PI / self.waist_y)
        self.zRy = 0.5 * self.k0 * self.waist_y**2  #  vertical Rayleigh range [m]
        self.qy0 = 0.0 + self.zRy * 1j

        return

    def set_z_waist_x(self, _zWaistX):
        """
        set longitudinal position of the horizontal waist [m]
        """
        self.z_waist_x = _zWaistX
        self.dzwx = _waistX - self.z_waist
        return

    def set_z_waist_y(self, _zWaistY):
        """
        set longitudinal position of the vertical waist [m]
        """
        self.z_waist_y = _zWaistY
        self.dzwy = _waistY - self.z_waist
        return

    def setMCoef(self, hCoefs):
        """
        set array of horizontal coefficients (complex)
        """
        self.mMax = hCoefs.size
        self.hCoefs = hCoefs
        return

    def setNCoef(self, vCoefs):
        """
        set array of vertical coefficients (complex)
        """
        self.nMax = vCoefs.size
        self.vCoefs = vCoefs
        return

    def setCoeffSingleModeX(self, mMode, mCoef):
        """
        set horiz. mode number & coeff for single mode
        """
        self.mMax = mMode + 1
        self.hCoefs = np.zeros(self.mMax) + np.zeros(self.mMax) * 1j
        self.hCoefs[mMode] = mCoef
        return

    def setCoeffSingleModeY(self, nMode, nCoef):
        """
        set vertical mode num. & coeff for single mode
        """
        self.nMax = nMode + 1
        self.vCoefs = np.zeros(self.nMax) + np.zeros(self.nMax) * 1j
        self.vCoefs[nMode] = nCoef
        return

    def evaluate_ex(self, xArray, yArray, _z, tArray):
        """
        For now, we assume this is the polarization direction

        Args:
            x,y,z,t can all be scalar, to evaluate at a single point.
            x,y can both be arrays (same length) to evaluate on a mesh.
            t can be an array, to evaluate at a sequence of times.
            x,y,t can all be arrays, for a particle distribution with fixed z
            _z, the longitudinal coordinate, must always be a scalar.
        """

        # account for location of pulse center
        z_local = _z - self.z_center

        # get the complex-valued envelope function
        result = self.evaluate_envelope_ex(xArray, yArray, _z)

        # multiply by the time-dependent term
        return result * np.exp((self.omega0 * tArray - self.k0 * z_local) * 1j)

    def evaluate_envelope_ex(self, xArray, yArray, _z):
        """
        For now, we assume this is the polarization direction

        Args:
            x,y,z can all be scalar, to evaluate at a single point.
            x,y can both be arrays (same length) to evaluate on a mesh.
            _z, the longitudinal coordinate, must always be a scalar.

        Note:
            We ignore x/y differences in the waist size and location
            Also, we ignore the higher-order Hermite modes here.
        """
        # account for location of pulse center
        z_local = _z - self.z_center
        # account for the waist location
        # _z -= self.z_waist

        # determine whether xArray is really a Numpy array
        try:
            num_vals_x = xArray.size
            x_is_array = True
        except AttributeError:
            # above failed, so input must be a float
            x_is_array = False

        # determine whether yArray is really a Numpy array
        try:
            num_vals_y = yArray.size
            y_is_array = True
        except AttributeError:
            # above failed, so input must be a float
            y_is_array = False

        if x_is_array and y_is_array:
            rSq = np.zeros(num_vals_x, complex)
            exp_1 = np.zeros(num_vals_x, complex)
            exp_2 = np.zeros(num_vals_x, complex)
            arg_2 = np.zeros(num_vals_x, complex)

        # radius at which the field amplitudes fall to exp(-1) of their axial values
        #     i.e., where the intensity values fall to exp(-2)
        wZ = self.w0 * math.sqrt(1 + (_z / self.zR) ** 2)
        #        pkdc('w(z)/w0 = ' + str(wZ/self.w0))

        # the radius squared
        rSq = np.power(xArray, 2) + np.power(yArray, 2)

        # the radius of curvature of wavefronts at location z
        invR = _z / (_z**2 + self.zR**2)

        # first exponential
        exp_1 = np.exp(-rSq / wZ**2)

        # Gouy phase at position z
        psi_z = np.arctan(_z / self.zR)

        # 2nd exponential
        arg_2 = 0.5 * self.k0 * invR * rSq
        exp_2 = np.exp(-1j * (arg_2 - psi_z))

        #        pkdc(' k0 = ' + str(self.k0))
        #        pkdc(' invR = ' + str(invR))
        #        pkdc(' rSq = ' + str(rSq))
        #        pkdc(' arg_2 = ' + str(arg_2))
        #        pkdc(' psi_z = ' + str(psi_z))
        #        pkdc(' Re[exp_2] = ' + str(np.real(exp_2)))

        # return the complex valued result
        # here, we apply a longitudinal Gaussian profile
        return (
            (self.w0 / wZ)
            * exp_1
            * np.exp(-((z_local / self.L_fwhm) ** 2))
            * self.efield0
            * exp_2
        )

    def evaluate_er(self, rArray, _z, tArray):
        """
        Evaluate the radial electric field of a circularly polarized laser pulse in r-z geometry.

        Args:
           rArray can be a scalar, to evaluate at a single point.
           rArray can be a Numpy array to evaluate along a line.
           _z, the longitudinal coordinate, must always be a scalar.
           tArray can be a scalar (works) or a Numpy array (not tested)
        """

        # account for location of pulse center
        z_local = _z - self.z_center

        # get the complex-valued envelope function
        result = self.evaluate_envelope_er(rArray, _z)

        # multiply by the time-dependent term
        return result * np.exp((self.omega0 * tArray - self.k0 * z_local) * 1j)

    def evaluate_envelope_er(self, rArray, _z):
        """
        Calculate the laser pulse envelope in radial r-z coordinates

        Args:
            rArray can be a scalar, to evaluate at a single point.
            rArray can be a Numpy array to evaluate along a line
            _z, the longitudinal coordinate, must always be a scalar.

        Note:
            We ignore x/y differences in the waist size and location
            Also, we ignore the higher-order Hermite modes here.
        """
        # account for the waist location
        z_local = _z - self.z_center
        _z -= self.z_waist
        #        pkdc('z_local, _z = ' + str(z_local) + ', ' + str(_z))

        # determine whether xArray is really a Numpy array
        try:
            num_vals_r = rArray.size
            r_is_array = True
            rSq = np.zeros(num_vals_r, complex)
            exp_1 = np.zeros(num_vals_r, complex)
            exp_2 = np.zeros(num_vals_r, complex)
            arg_2 = np.zeros(num_vals_r, complex)
        except AttributeError:
            # above failed, so input must be a float
            r_is_array = False

        # radius at which the field amplitudes fall to exp(-1) of their axial values
        #     i.e., where the intensity values fall to exp(-2)
        wZ = self.w0 * math.sqrt(1 + (_z / self.zR) ** 2)
        #        pkdc('w(z)/w0 = ' + str(wZ/self.w0))

        # the radius squared
        rSq = np.power(rArray, 2)

        # the radius of curvature of wavefronts at location z
        invR = _z / (_z**2 + self.zR**2)

        # first exponential
        exp_1 = np.exp(-rSq / wZ**2)

        # Gouy phase at position z
        psi_z = np.arctan(_z / self.zR)

        # 2nd exponential
        arg_2 = 0.5 * self.k0 * invR * rSq - psi_z
        exp_2 = np.exp(-1j * arg_2)

        #        pkdc(' k0 = ' + str(self.k0))
        #        pkdc(' invR = ' + str(invR))
        #        pkdc(' rSq = ' + str(rSq))
        #        pkdc(' arg_2 min/max = ' + str(np.min(arg_2)) + ', ' + str(np.max(arg_2)))
        #        pkdc(' psi_z = ' + str(psi_z))
        #        pkdc(' Re[exp_2] = ' + str(np.real(exp_2)))

        # return the complex valued result
        # here, we apply a longitudinal Gaussian profile
        return (
            (self.w0 / wZ)
            * exp_1
            * np.exp(-((z_local / self.L_fwhm) ** 2))
            * self.efield0
            * exp_2
        )

    def eval_gh_ex(self, xArray, yArray, z):
        """
        Note: This is old, untested code with no concept of a longitudinal envelope
        The complicated logic requires further testing.

        Args:
            x,y,z can all be scalar, to evaluate at a single point.
            x,y can both be arrays (same length) to evaluate on a mesh.
            z, the longitudinal coordinate, must always be a scalar.
        """

        # assume array input; try to create temporary array
        try:
            numVals = xArray.size
            result = np.zeros(numVals, complex)
        except AttributeError:
            # above failed, so input must be a float
            result = 0.0

        # rotate and shift the x,y coordinates as necessary
        rotArg = (xArray - self.x_shift) * math.cos(self.wRotAngle) + (
            yArray - self.y_shift
        ) * math.sin(self.wRotAngle)

        # z-dependent temporary variables
        qxz = (z - self.z_waist_x) + self.qx0
        xrFac = 0.5 * self.k0 / qxz
        xzFac = (
            math.sqrt(2)
            / self.waist_x
            / math.sqrt(1 + ((z - self.z_waist_x) / self.zRx) ** 2)
        )

        # load up array of mode-dependent factors
        xCoefs = np.zeros(self.mMax, complex)
        for mMode in range(self.mMax):
            xCoefs[mMode] = (
                self.hCoefs[mMode]
                * self.piWxFac
                * cmath.sqrt(self.qx0 / qxz)
                / math.sqrt(math.factorial(mMode) * (2.0**mMode))
                * (self.qx0 * qxz.conjugate() / self.qx0.conjugate() / qxz)
                ** (0.5 * mMode)
            )

        # evaluate the product of Hermite series
        result = hermval(xzFac * rotArg, xCoefs)
        result *= np.exp(-(xrFac * rotArg**2) * 1j)

        #
        # rinse and repeat:  do the same for the y-dependent Hermite series --
        #

        # rotate and shift the x,y coordinates as necessary
        rotArg = (yArray - self.y_shift) * math.cos(self.wRotAngle) - (
            xArray - self.x_shift
        ) * math.sin(self.wRotAngle)

        # z-dependent temporary variables
        qyz = (z - self.z_waist_y) + self.qy0
        yrFac = 0.5 * self.k0 / qyz
        yzFac = (
            math.sqrt(2)
            / self.waist_y
            / math.sqrt(1 + ((z - self.z_waist_y) / self.zRy) ** 2)
        )

        # load up array of mode-dependent factors
        xCoefs = np.zeros(self.mMax, complex)
        for mMode in range(self.mMax):
            xCoefs[mMode] = (
                self.hCoefs[mMode]
                * self.piWxFac
                * cmath.sqrt(self.qx0 / qxz)
                / math.sqrt(math.factorial(mMode) * (2.0**mMode))
                * (self.qx0 * qxz.conjugate() / self.qx0.conjugate() / qxz)
                ** (0.5 * mMode)
            )

        # load up array of mode-dependent factors (y-dependence)
        yCoefs = np.zeros(self.nMax, complex)
        for nMode in range(self.nMax):
            yCoefs[nMode] = (
                self.vCoefs[nMode]
                * self.piWyFac
                * cmath.sqrt(self.qy0 / qyz)
                / math.sqrt(math.factorial(nMode) * (2.0**nMode))
                * (self.qy0 * qyz.conjugate() / self.qy0.conjugate() / qyz)
                ** (0.5 * nMode)
            )

        # evaluate product of Hermite series (multiplying into previous value)
        result *= hermval(yzFac * rotArg, yCoefs)
        result *= np.exp(-(yrFac * rotArg**2) * 1j)

        # return the complex valued result
        return result


def _nan_helper(_arr):
    """
    Clean unwanted NaNs from a numpy array, replacing them via interpolation.

    Args:
        _arr, numpy array with NaNs

    Returns:
        nans, logical indices of NaNs
        index, a function with signature indices = index(logical_indices)
               to convert logical indices of NaNs to 'equivalent' indices

    Example:
        >>> nans, x = nan_helper(my_array)
        >>> my_array[nans] = np.interp(x(nans), x(~nans), my_array[~nans])
    """
    return np.isnan(_arr), lambda z: z.nonzero()[0]


def _array_cleaner(_arr, _ind):
    """
    Clean unwanted values from a numpy array, replacing them via interpolation.

    Args:
        _arr, numpy array with bad values
        _ind, precalculated indices of these bad values

    Returns:
        _arr, cleaned version of the input array

    Example:
        >>> indices = np.isnan(my_array)
        >>> my_array = array_cleaner(my_array, indices)
    """
    _arr[_ind] = np.nan
    nans, x = _nan_helper(_arr)
    _arr[nans] = np.interp(x(nans), x(~nans), _arr[~nans])
    return _arr


def gaussian_pad(data):
    # Takes a 2d array, fits a gaussian to the non-zero values,
    #  and replaces the zero values with their respective value from the fit,
    #  then it smooths the result

    # Code taken from examples/smoothing/gaussian_01
    def gaussian(params, amplitude, xo, yo, sigma):
        xo = float(xo)
        yo = float(yo)
        r = np.sqrt((params[0] - xo) ** 2 + (params[1] - yo) ** 2)
        g = amplitude * np.exp(-((r / sigma) ** 2))
        return g.ravel()

    x = np.linspace(0, data.shape[1] - 1, data.shape[1])
    y = np.linspace(0, data.shape[0] - 1, data.shape[0])
    x, y = np.meshgrid(x, y)

    initial_guess = (np.max(data), len(x), len(y), 10)
    popt, pcov = opt.curve_fit(
        gaussian, (x, y), data.flatten(), p0=initial_guess, maxfev=10000
    )
    data_fit = gaussian((x, y), *popt).reshape(data.shape)

    data_new = np.copy(data)
    data_new[np.where(data == 0)] = data_fit[np.where(data == 0)]

    # Smooth array
    blur = 2
    data_new_smooth = gaussian_filter(data_new, sigma=blur)

    return data_new_smooth


def _replace_phase_nan(wfs_data):
    x = np.linspace(0, np.shape(wfs_data)[1] - 1, np.shape(wfs_data)[1])
    y = np.linspace(0, np.shape(wfs_data)[0] - 1, np.shape(wfs_data)[0])
    center = np.argwhere(wfs_data == np.max(wfs_data[~np.isnan(wfs_data)]))[0]
    x_shifted = np.copy(x) - center[1]
    y_shifted = np.copy(y) - center[0]

    nan_indices = np.argwhere(np.isnan(wfs_data))
    nan_r = np.sqrt(
        x_shifted[nan_indices[:, 1]] ** 2.0 + y_shifted[nan_indices[:, 0]] ** 2.0
    )
    new_nan_indices = nan_indices[np.argsort(nan_r)]

    for x_index, y_index in new_nan_indices:
        x_temp = np.copy(x) - x[y_index]
        y_temp = np.copy(y) - y[x_index]
        xv_temp, yv_temp = np.meshgrid(x_temp, y_temp)
        r_temp = np.sqrt(xv_temp**2.0 + yv_temp**2.0)
        nan_location = r_temp[x_index, y_index]

        closest_phase_data = np.zeros(np.shape(r_temp)) + 0.3
        closest_phase_data[~np.isnan(wfs_data)] = wfs_data[~np.isnan(wfs_data)]

        r_temp[np.isnan(wfs_data)] = np.max(r_temp) * 2.0

        closest_indices = np.argwhere(
            np.abs(r_temp - nan_location) == np.min(np.abs(r_temp - nan_location))
        )
        wfs_data[x_index, y_index] = np.mean(
            closest_phase_data[closest_indices[:, 0], closest_indices[:, 1]]
        )

    return wfs_data


def bin_arrays(y_array, x_array, n_values):

    value_max = np.max(x_array)
    value_min = np.min(x_array)

    x_array_values = np.linspace(value_min, value_max, n_values)
    x_array_bins = np.append(
        np.copy(x_array_values), x_array_values[-1] + np.diff(x_array_values)[0]
    ) - (np.diff(x_array_values)[0] / 2.0)

    dbin = np.diff(x_array_bins)[0]
    x_width = dbin

    y_array_binned = np.zeros(np.shape(x_array_values))

    digitized_x_array = np.digitize(x_array, x_array_bins)
    for bin_index in range(1, len(x_array_bins)):

        front_bin_value = x_array_bins[bin_index - 1]
        end_bin_value = x_array_bins[bin_index]

        loc_in_this_bin = np.where(digitized_x_array == bin_index)

        x_values_in_this_bin = x_array[loc_in_this_bin]
        y_values_in_this_bin = y_array[loc_in_this_bin]

        front_half = np.where(x_values_in_this_bin < front_bin_value + (dbin / 2.0))
        end_half = np.where(x_values_in_this_bin >= front_bin_value + (dbin / 2.0))

        fraction_here_fh = (
            (x_values_in_this_bin[front_half] + (x_width / 2.0)) - front_bin_value
        ) / dbin
        fraction_before_fh = (
            front_bin_value - (x_values_in_this_bin[front_half] - (x_width / 2.0))
        ) / dbin

        fraction_here_eh = (
            end_bin_value - (x_values_in_this_bin[end_half] - (x_width / 2.0))
        ) / dbin
        fraction_after_eh = (
            (x_values_in_this_bin[end_half] + (x_width / 2.0)) - end_bin_value
        ) / dbin

        y_array_binned[bin_index - 1] += np.sum(
            fraction_here_fh * y_values_in_this_bin[front_half]
        )
        try:
            y_array_binned[bin_index - 2] += np.sum(
                fraction_before_fh * y_values_in_this_bin[front_half]
            )
        except:
            y_array_binned[bin_index - 1] += np.sum(
                fraction_before_fh * y_values_in_this_bin[front_half]
            )

        y_array_binned[bin_index - 1] += np.sum(
            fraction_here_eh * y_values_in_this_bin[end_half]
        )
        try:
            y_array_binned[bin_index] += np.sum(
                fraction_after_eh * y_values_in_this_bin[end_half]
            )
        except:
            y_array_binned[bin_index - 1] += np.sum(
                fraction_after_eh * y_values_in_this_bin[end_half]
            )

    return x_array_values, y_array_binned
