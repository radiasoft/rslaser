# -*- coding: utf-8 -*-
"""Definition of a laser pulse
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import array
import math
import cmath
import numpy as np
from pykern.pkdebug import pkdp, pkdlog
from pykern.pkcollections import PKDict
from numpy.polynomial.hermite import hermval
import rslaser.optics.wavefront as rswf
import rsmath.const as rsc
import rslaser.utils.unit_conversion as units
import rslaser.utils.srwl_uti_data as srwutil
import scipy.constants as const
import scipy.ndimage as ndi
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter
from scipy import special
from scipy import signal
from skimage import filters
from skimage import img_as_float
import scipy.optimize as opt
import srwlib
from srwlib import srwl
from rslaser.utils.validator import ValidatorBase
import matplotlib.pyplot as plt

_LASER_PULSE_DEFAULTS = PKDict(
    nslice=3,
    pulse_direction=0.0, #0 corresponds to 'right' or 'z' or 'forward', can be set to any angle relative
    chirp=0,
    photon_e_ev=1.5,  # 1e3,
    num_sig_long=3.0,
    dist_waist=0,
    tau_fwhm=0.1 / const.c / math.sqrt(2.0),
    pulseE=0.001,
    sigx_waist=1.0e-3,
    sigy_waist=1.0e-3,
    num_sig_trans=6,
    nx_slice=64,
    ny_slice=64,
    poltype=1,
    mx=0,
    my=0,
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
    photon_e_ev=1.5,  # 1e3,
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
                tau_fwhm (float): FWHM laser pulse length [s]
                pulseE (float): total laser pulse energy [J]
                z_center (float): # longitudinal location of pulse center [m]
                x_shift (float): horizontal shift of the spot center [m]
                y_shift (float): vertical shift of the spot center [m]
                sigx_waist (float): horizontal RMS waist size [m]
                sigy_waist (float): vertical RMS waist size [m]
                nx_slice (int): no. of horizontal mesh points in slice
                ny_slice (int): no. of vertical mesh points in slice
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
        # self.photon_e -= 0.5*params.chirp           # so central slice has the central photon energy
        # _de = params.chirp / self.nslice   # photon energy shift from slice to slice
        for i in range(params.nslice):
            # add the slices; each (slowly) instantiates an SRW wavefront object
            self.slice.append(LaserPulseSlice(i, params.copy(), files=self.files))
        self._sxvals = []  # horizontal slice data
        self._syvals = []  # vertical slice data

    def resize_laser_mesh(self):
        # Manually force mesh back to original extent + number of cells

        for laser_index_i in np.arange(self.nslice):
            thisSlice = self.slice[laser_index_i]

            new_x = np.linspace(
                thisSlice.wfr.mesh.xStart,
                thisSlice.wfr.mesh.xFin,
                thisSlice.wfr.mesh.nx,
            )
            new_y = np.linspace(
                thisSlice.wfr.mesh.yStart,
                thisSlice.wfr.mesh.yFin,
                thisSlice.wfr.mesh.ny,
            )

            # Check if need to make changes
            if not (
                np.array_equal(new_x, thisSlice.initial_laser_xy.x)
                and np.array_equal(new_y, thisSlice.initial_laser_xy.y)
            ):

                re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d = srwutil.extract_2d_fields(
                    thisSlice.wfr
                )

                # Interpolate ex meshes
                rect_biv_spline_xre = RectBivariateSpline(new_x, new_y, re_ex_2d)
                rect_biv_spline_xim = RectBivariateSpline(new_x, new_y, im_ex_2d)
                new_re_ex_2d = rect_biv_spline_xre(
                    thisSlice.initial_laser_xy.x, thisSlice.initial_laser_xy.y
                )
                new_im_ex_2d = rect_biv_spline_xim(
                    thisSlice.initial_laser_xy.x, thisSlice.initial_laser_xy.y
                )

                # Interpolate ey meshes
                rect_biv_spline_yre = RectBivariateSpline(new_x, new_y, re_ey_2d)
                rect_biv_spline_yim = RectBivariateSpline(new_x, new_y, im_ey_2d)
                new_re_ey_2d = rect_biv_spline_yre(
                    thisSlice.initial_laser_xy.x, thisSlice.initial_laser_xy.y
                )
                new_im_ey_2d = rect_biv_spline_yim(
                    thisSlice.initial_laser_xy.x, thisSlice.initial_laser_xy.y
                )

                thisSlice.wfr = srwutil.make_wavefront(
                    new_re_ex_2d,
                    new_im_ex_2d,
                    new_re_ey_2d,
                    new_im_ey_2d,
                    thisSlice.photon_e_ev,
                    thisSlice.initial_laser_xy.x,
                    thisSlice.initial_laser_xy.y,
                )

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

        for laser_index_i in np.arange(nslices_pulse):

            thisSlice = self.slice[laser_index_i]
            slice_wfr = thisSlice.wfr

            # total component of electric field
            re0, re0_mesh = srwutil.calc_int_from_wfr(
                slice_wfr, _pol=6, _int_type=5, _det=None, _fname="", _pr=False
            )
            im0, im0_mesh = srwutil.calc_int_from_wfr(
                slice_wfr, _pol=6, _int_type=6, _det=None, _fname="", _pr=False
            )

            e_total.re[:, :, laser_index_i] = (
                np.array(re0)
                .reshape((slice_wfr.mesh.nx, slice_wfr.mesh.ny), order="C")
                .astype(np.float64)
            )
            e_total.im[:, :, laser_index_i] = (
                np.array(im0)
                .reshape((slice_wfr.mesh.nx, slice_wfr.mesh.ny), order="C")
                .astype(np.float64)
            )

            # gaussian scale
            slice_end1 = (thisSlice._pulse_pos - 0.5 * thisSlice.ds) / (
                2.0 * thisSlice.sig_s
            )
            slice_end2 = (thisSlice._pulse_pos + 0.5 * thisSlice.ds) / (
                2.0 * thisSlice.sig_s
            )
            slice_width_factor = (
                1.0
                / np.exp(-thisSlice._pulse_pos**2.0 / (2.0 * thisSlice.sig_s) ** 2.0)
            ) * ((special.erf(slice_end2) - special.erf(slice_end1)) / pulse_factor)

            # scale slice fields by factor to represent full slice, not just middle value
            e_total.re[:, :, laser_index_i] *= slice_width_factor
            e_total.im[:, :, laser_index_i] *= slice_width_factor

        e_total.re = np.sum(e_total.re, axis=2)
        e_total.im = np.sum(e_total.im, axis=2)

        return e_total

    def combine_n2_variation(self, laser_pulse_copies, cut_offs, max_n2):

        for laser_index_i in np.arange(self.nslice):
            # For each laser slice, combine the propagated wavefronts
            wfr_max = laser_pulse_copies.n2_max.slice[laser_index_i].wfr
            wfr_0 = laser_pulse_copies.n2_0.slice[laser_index_i].wfr

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
                n2_max=np.unwrap(
                    np.unwrap(
                        np.array(phase_1d_max)
                        .reshape((wfr_max.mesh.nx, wfr_max.mesh.ny), order="C")
                        .astype(np.float64),
                        axis=0,
                    ),
                    axis=1,
                ),
                n2_0=np.unwrap(
                    np.unwrap(
                        np.array(phase_1d_0)
                        .reshape((wfr_0.mesh.nx, wfr_0.mesh.ny), order="C")
                        .astype(np.float64),
                        axis=0,
                    ),
                    axis=1,
                ),
            )

            # identify radial distance of every cell
            x = np.linspace(wfr_0.mesh.xStart, wfr_0.mesh.xFin, wfr_0.mesh.nx)
            y = np.linspace(wfr_0.mesh.yStart, wfr_0.mesh.yFin, wfr_0.mesh.ny)
            xv, yv = np.meshgrid(x, y)
            r = np.sqrt(xv**2.0 + yv**2.0)

            x_loc = cut_offs[1]
            y_loc = int(len(y) / 2.0)
            shift_value = (
                phase_2d.n2_max[x_loc - 1, y_loc] - phase_2d.n2_0[x_loc, y_loc]
            )
            phase_2d.n2_0 += shift_value

            # Assign the n2 = 0 fields initially
            intensity = intensity_2d.n2_0
            phase = phase_2d.n2_0

            n2 = np.zeros(np.shape(intensity))
            self.slice[laser_index_i].n_photons_2d.mesh = laser_pulse_copies.n2_0.slice[
                laser_index_i
            ].n_photons_2d.mesh

            temp_x = np.linspace(0.0, 1.0, len(x[int(len(x) / 2.0) + 1 : cut_offs[1]]))
            a = 1.0 - temp_x**2.0

            def scaling_fn(index, array_1, array_2):
                return ((1.0 - a[index]) * array_1) + (a[index] * array_2)

            for index, value in enumerate(
                np.flip(x[int(len(x) / 2.0) + 1 : cut_offs[1]])
            ):
                location = np.where(r <= value)
                n2[location] = scaling_fn(index, max_n2, 0.0)
                intensity[location] = scaling_fn(
                    index, intensity_2d.n2_max[location], intensity_2d.n2_0[location]
                )
                phase[location] = scaling_fn(
                    index, phase_2d.n2_max[location], phase_2d.n2_0[location]
                )
            
            e_norm = np.sqrt(2.0 * intensity / (const.c * const.epsilon_0))
            re_ex = np.multiply(e_norm, np.cos(phase))
            im_ex = np.multiply(e_norm, np.sin(phase))
            re_ey = np.zeros(np.shape(re_ex))
            im_ey = np.zeros(np.shape(im_ex))

            # remake the wavefront
            self.slice[laser_index_i].wfr = srwutil.make_wavefront(
                re_ex, im_ex, re_ey, im_ey, self.slice[laser_index_i].photon_e_ev, x, y
            )

        return self

    def ideal_mirror_180(self):
        # 0.0 corresponds to 'right' or 'x=0,y=0,z' or 'forward'
        if self.pulse_direction==0.0:
            self.pulse_direction = 180.0
        elif self.pulse_direction==180.0:
            self.pulse_direction = 0.0
        
        for laser_index_i in np.arange(self.nslice):
            thisSlice = self.slice[laser_index_i]

            re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d = srwutil.extract_2d_fields(
                thisSlice.wfr
            )
        
            # E_f (x,y) = E_i (-x,-y)
            new_re_ex_2d = np.flip(re_ex_2d)
            new_im_ex_2d = np.flip(im_ex_2d)
            new_re_ey_2d = np.flip(re_ey_2d)
            new_im_ey_2d = np.flip(im_ey_2d)
            
            thisSlice.wfr = srwutil.make_wavefront(
                new_re_ex_2d,
                new_im_ex_2d,
                new_re_ey_2d,
                new_im_ey_2d,
                thisSlice.photon_e_ev,
                thisSlice.initial_laser_xy.x,
                thisSlice.initial_laser_xy.y,
            )

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
        # self.z_waist = params.z_waist
        self.nslice = params.nslice
        self.nx_slice = params.nx_slice
        self.ny_slice = params.ny_slice
        self.dist_waist = params.dist_waist

        #  (Note KW: called this pulseE_slice because right now LPS is also passed pulseE for the whole pulse)
        self.pulseE_slice = (
            params.pulseE / self.nslice
        )  # currently assumes consistent length and energy across all slices

        # compute slice photon energy from central energy, chirp, and slice index
        self.photon_e_ev = (
            params.photon_e_ev
        )  # check that this is being properly incremented in the correct place (see LaserPulse class)
        _de = params.chirp / self.nslice  # photon energy shift from slice to slice
        self.photon_e_ev -= 0.5 * params.chirp + (
            self.nslice * _de
        )  # so central slice has the central photon energy

        self.sig_s = params.tau_fwhm * const.c / 2.355
        self.num_sig_long = params.num_sig_long
        constConvRad = 1.23984186e-06 / (
            4 * 3.1415926536
        )  ##conversion from energy to 1/wavelength
        rmsAngDiv_x = constConvRad / (
            self.photon_e_ev * self.sigx_waist
        )  ##RMS angular divergence [rad]
        rmsAngDiv_y = constConvRad / (self.photon_e_ev * self.sigy_waist)

        sigrL_x = math.sqrt(self.sigx_waist**2 + (self.dist_waist * rmsAngDiv_x) ** 2)
        sigrL_y = math.sqrt(self.sigy_waist**2 + (self.dist_waist * rmsAngDiv_y) ** 2)

        # *************begin function below**********

        # sig_s = params.tau_fwhm * const.c / 2.355
        self.ds = (
            2 * params.num_sig_long * self.sig_s / params.nslice
        )  # longitudinal spacing between slices
        # self._pulse_pos = self.dist_waist - params.num_sig_long * self.sig_s + (slice_index + 0.5) * self.ds
        self._pulse_pos = (
            -params.num_sig_long * self.sig_s + (slice_index + 0.5) * self.ds
        )
        self._wavefront(params, files)

        # Calculate the initial number of photons in 2d grid of each slice from pulseE_slice
        self.n_photons_2d = self.calc_init_n_photons()  # 2d array

        self.initial_laser_xy = PKDict(
            x=np.linspace(self.wfr.mesh.xStart, self.wfr.mesh.xFin, self.wfr.mesh.nx),
            y=np.linspace(self.wfr.mesh.yStart, self.wfr.mesh.yFin, self.wfr.mesh.ny),
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
            indices = np.isnan(wfs_data)
            wfs_data = _array_cleaner(wfs_data, indices)

            nx_wfs = np.shape(wfs_data)[0]
            ny_wfs = np.shape(wfs_data)[1]
            nx_ccd = np.shape(ccd_data)[0]
            ny_ccd = np.shape(ccd_data)[1]

            # Increase the shape to 64x64: pad wfs data with array edge, pad ccd data with zeros
            if nx_wfs < 64:
                wfs_data = np.pad(
                    wfs_data,
                    ((int((64 - nx_wfs) / 2), int((64 - nx_wfs) / 2)), (0, 0)),
                    mode="edge",
                )
            if ny_wfs < 64:
                wfs_data = np.pad(
                    wfs_data,
                    ((0, 0), (int((64 - ny_wfs) / 2), int((64 - ny_wfs) / 2))),
                    mode="edge",
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
            self.wfs_norm_factor = 2841.7370456965646

            # scale for slice location
            number_slices_correction = np.exp(
                -self._pulse_pos**2.0 / (2.0 * self.sig_s) ** 2.0
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
        constant_factor = 7.3948753166511745
        length_factor = constant_factor / self.ds

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
            self.ny_slice,
            self.photon_e_ev,
            params.mx,
            params.my,
        )

    def calc_init_n_photons(self):

        # Note: assumes longitudinal gaussian profile when initializing

        # intensity = srwlib.array('f', [0]*self.wfr.mesh.nx*self.wfr.mesh.ny) # "flat" array to take 2D intensity data
        # srwl.CalcIntFromElecField(intensity, self.wfr, 0, 0, 3, self.wfr.mesh.eStart, 0, 0) #extracts intensity

        # # Reshaping intensity data from flat to 2D array
        # intens_2d = np.array(intensity).reshape((self.wfr.mesh.nx, self.wfr.mesh.ny), order='C').astype(np.float64)

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
