# -*- coding: utf-8 -*-
"""Definition of a crystal
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import numpy as np
import array
import math
import copy
from pykern.pkcollections import PKDict
import srwlib
import scipy.constants as const
from scipy.interpolate import RectBivariateSpline
from rsmath import lct as rslct
from rslaser.utils.validator import ValidatorBase
from rslaser.utils import srwl_uti_data as srwutil
from rslaser.optics.element import ElementException, Element

_N_SLICE_DEFAULT = 50
_N0_DEFAULT = 1.75
_N2_DEFAULT = 0.001
_CRYSTAL_DEFAULTS = PKDict(
    n0=[_N0_DEFAULT for _ in range(_N_SLICE_DEFAULT)],
    n2=[_N2_DEFAULT for _ in range(_N_SLICE_DEFAULT)],
    length=0.2,
    l_scale=1,
    nslice=_N_SLICE_DEFAULT,
    slice_index=0,
    # A = 9.99988571e-01,
    # B = 1.99999238e-01,
    # C = -1.14285279e-04,
    # D = 9.99988571e-01,
    A=0.99765495,
    B=1.41975385,
    C=-0.0023775,
    D=0.99896716,
    population_inversion=PKDict(
        n_cells=64,
        mesh_extent=0.01,  # [m]
        crystal_alpha=120.0,  # [1/m], 1.2 1/cm
        pump_waist=0.00164,  # [m]
        pump_wavelength=532.0e-9,  # [m]
        pump_energy=0.0211,  # [J], pump laser energy onto the crystal
        pump_type="dual",
    ),
)


class Crystal(Element):
    """
    Args:
        params (PKDict) with fields:
            n0 (float): array of on axis index of refractions in crystal slices
            n2 (float): array of quadratic variations of index of refractions, with n(r) = n0 - 1/2 n2 r^2  [m^-2]
            note: n0, n2 should be an array of length nslice; if nslice = 1, they should be single values
            length (float): total length of crystal [m]
            nslice (int): number of crystal slices
            l_scale: length scale factor for LCT propagation
    """

    _DEFAULTS = _CRYSTAL_DEFAULTS
    _INPUT_ERROR = ElementException

    def __init__(self, params=None):
        params = self._get_params(params)
        self._validate_params(params)

        # Check if n2<0, throw an exception if true
        if (np.array(params.n2) < 0.0).any():
            raise self._INPUT_ERROR(f"You've specified negative value(s) for n2")

        self.length = params.length
        self.nslice = params.nslice
        self.l_scale = params.l_scale
        self.slice = []
        for j in range(self.nslice):
            p = params.copy()
            p.update(
                PKDict(
                    n0=params.n0[j],
                    n2=params.n2[j],
                    length=params.length / params.nslice,
                    slice_index=j,
                )
            )
            self.slice.append(CrystalSlice(params=p))

    def _get_params(self, params):
        def _update_n0_and_n2(params_final, params, field):
            if len(params_final[field]) != params_final.nslice:
                if not params.get(field):
                    # if no n0/n2 specified then we use default nlice times in array
                    params_final[field] = [
                        PKDict(
                            n0=_N0_DEFAULT,
                            n2=_N2_DEFAULT,
                        )[field]
                        for _ in range(params_final.nslice)
                    ]
                    return
                raise self._INPUT_ERROR(
                    f"you've specified an {field} unequal length to nslice"
                )

        o = params.copy() if type(params) == PKDict else PKDict()
        p = super()._get_params(params)
        if not o.get("nslice") and not o.get("n0") and not o.get("n2"):
            # user specified nothing, use defaults provided by _get_params
            return p
        if o.get("nslice"):
            # user specifed nslice, but not necissarily n0/n2
            _update_n0_and_n2(p, o, "n0")
            _update_n0_and_n2(p, o, "n2")
            return p
        if o.get("n0") or o.get("n2"):
            if len(p.n0) < p.nslice or len(p.n2) < p.nslice:
                p.nslice = min(len(p.n0), len(p.n2))
        return p

    def propagate(self, laser_pulse, prop_type, calc_gain=False, radial_n2=False):
        assert (laser_pulse.pulse_direction == 0.0) or (
            laser_pulse.pulse_direction == 180.0
        ), "ERROR -- Propagation not implemented for the pulse direction {}".format(
            laser_pulse.pulse_direction
        )

        if laser_pulse.pulse_direction == 0.0:
            slice_array = self.slice
        elif laser_pulse.pulse_direction == 180.0:
            slice_array = self.slice[::-1]

        for s in slice_array:

            if radial_n2:

                assert prop_type == "n0n2_srw", "ERROR -- Only implemented for n0n2_srw"
                laser_pulse_copies = PKDict(
                    n2_max=copy.deepcopy(laser_pulse),
                    n2_0=copy.deepcopy(laser_pulse),
                )

                temp_crystal_slice = copy.deepcopy(s)
                temp_crystal_slice.n2 = 0.0

                laser_pulse_copies.n2_max = s.propagate(
                    laser_pulse_copies.n2_max, prop_type, calc_gain
                )
                laser_pulse_copies.n2_0 = temp_crystal_slice.propagate(
                    laser_pulse_copies.n2_0, prop_type, calc_gain
                )

                x = np.linspace(
                    laser_pulse.slice[0].wfr.mesh.xStart,
                    laser_pulse.slice[0].wfr.mesh.xFin,
                    laser_pulse.slice[0].wfr.mesh.nx,
                )

                zero_cut_off = (
                    1.3 * s.population_inversion.pump_waist
                )  # Outside this value is zero n2
                linear_cut_off = s.population_inversion.pump_waist / np.sqrt(
                    2.0
                )  # This value and within is linear n2 (if used)
                linear_index = (np.abs(x - linear_cut_off)).argmin()
                zero_index = (np.abs(x - zero_cut_off)).argmin()

                cut_offs = np.array([linear_index, zero_index])
                laser_pulse = laser_pulse.combine_n2_variation(
                    laser_pulse_copies, cut_offs, s.n2
                )
            else:
                laser_pulse = s.propagate(laser_pulse, prop_type, calc_gain)

            laser_pulse.resize_laser_mesh()
            laser_pulse.flatten_phase_edges()
        return laser_pulse


class CrystalSlice(Element):
    """
    This class represents a slice of a crystal in a laser cavity.

    Args:
        params (PKDict) with fields:
            length
            n0 (float): on-axis index of refraction
            n2 (float): transverse variation of index of refraction [1/m^2]
            n(r) = n0 - 0.5 n2 r^2
            l_scale: length scale factor for LCT propagation

    To be added: alpha0, alpha2 laser gain parameters

    Note: Initially, these parameters are fixed. Later we will update
    these parameters as the laser passes through.
    """

    _DEFAULTS = _CRYSTAL_DEFAULTS
    _INPUT_ERROR = ElementException

    def __init__(self, params=None):
        params = self._get_params(params)
        self._validate_params(params)
        self.length = params.length
        self.slice_index = params.slice_index
        self.n0 = params.n0
        self.n2 = params.n2
        self.l_scale = params.l_scale
        # self.pop_inv = params._pop_inv
        self.A = params.A
        self.B = params.B
        self.C = params.C
        self.D = params.D

        #  Assuming wfr0 exsts, created e.g. via
        #  wfr0=createGsnSrcSRW(sigrW,propLen,pulseE,poltype,photon_e_ev,sampFact,mx,my)
        # n_x = wfr0.mesh.nx  #  nr of grid points in x
        # n_y = wfr0.mesh.ny  #  nr of grid points in y
        # sig_cr_sec = np.ones((n_x, n_y), dtype=np.float32)

        # 2d mesh of excited state density (sigma)
        self._initialize_excited_states_mesh(params.population_inversion, params.nslice)

    def _left_pump(self, nslice, xv, yv):

        # z = distance from left of crystal to center of current slice (assumes all crystal slices have same length)
        z = self.length * (self.slice_index + 0.5)

        slice_front = z - (self.length / 2.0)
        slice_end = z + (self.length / 2.0)

        # calculate correction factor for representing a gaussian pulse with a series of flat-top slices
        correction_factor = (
            (
                np.exp(-self.population_inversion.crystal_alpha * slice_front)
                - np.exp(-self.population_inversion.crystal_alpha * slice_end)
            )
            / self.population_inversion.crystal_alpha
        ) / (np.exp(-self.population_inversion.crystal_alpha * z) * self.length)

        # Create a default mesh of [num_excited_states/m^3]
        pop_inversion_mesh = (
            (self.population_inversion.pump_wavelength / (const.h * const.c))
            * (
                (
                    2.0
                    * (
                        1
                        - np.exp(
                            -self.population_inversion.crystal_alpha
                            * self.length
                            * nslice
                        )
                    )
                    * (2.0 / 3.0)
                    * self.population_inversion.pump_energy
                    * np.exp(
                        -2.0
                        * (xv**2.0 + yv**2.0)
                        / self.population_inversion.pump_waist**2.0
                    )
                )
                / (const.pi * self.population_inversion.pump_waist**2.0)
            )
            * np.exp(-self.population_inversion.crystal_alpha * z)
            * correction_factor
        ) / (self.length * nslice)

        return pop_inversion_mesh

    def _right_pump(self, nslice, xv, yv):

        # z = distance from right of crystal to center of current slice (assumes all crystal slices have same length)
        z = self.length * ((nslice - self.slice_index - 1) + 0.5)

        slice_front = z - (self.length / 2.0)
        slice_end = z + (self.length / 2.0)

        # calculate correction factor for representing a gaussian pulse with a series of flat-top slices
        correction_factor = (
            (
                np.exp(-self.population_inversion.crystal_alpha * slice_front)
                - np.exp(-self.population_inversion.crystal_alpha * slice_end)
            )
            / self.population_inversion.crystal_alpha
        ) / (np.exp(-self.population_inversion.crystal_alpha * z) * self.length)

        # Create a default mesh of [num_excited_states/m^3]
        pop_inversion_mesh = (
            (self.population_inversion.pump_wavelength / (const.h * const.c))
            * (
                (
                    2.0
                    * (
                        1
                        - np.exp(
                            -self.population_inversion.crystal_alpha
                            * self.length
                            * nslice
                        )
                    )
                    * (2.0 / 3.0)
                    * self.population_inversion.pump_energy
                    * np.exp(
                        -2.0
                        * (xv**2.0 + yv**2.0)
                        / self.population_inversion.pump_waist**2.0
                    )
                )
                / (const.pi * self.population_inversion.pump_waist**2.0)
            )
            * np.exp(-self.population_inversion.crystal_alpha * z)
            * correction_factor
        ) / (self.length * nslice)

        return pop_inversion_mesh

    def _dual_pump(self, nslice, xv, yv):
        left_pump_mesh = self._left_pump(nslice, xv, yv)
        right_pump_mesh = self._right_pump(nslice, xv, yv)
        return left_pump_mesh + right_pump_mesh

    def _initialize_excited_states_mesh(self, population_inversion, nslice):
        self.population_inversion = population_inversion
        x = np.linspace(
            -population_inversion.mesh_extent,
            population_inversion.mesh_extent,
            population_inversion.n_cells,
        )
        xv, yv = np.meshgrid(x, x)

        self.pop_inversion_mesh = PKDict(
            dual=self._dual_pump,
            left=self._left_pump,
            right=self._right_pump,
        )[population_inversion.pump_type](nslice, xv, yv)

    def _propagate_attenuate(self, laser_pulse, calc_gain):
        # n_x = wfront.mesh.nx  #  nr of grid points in x
        # n_y = wfront.mesh.ny  #  nr of grid points in y
        # sig_cr_sec = np.ones((n_x, n_y), dtype=np.float32)
        # pop_inv = self.pop_inv
        # n0_phot = 0.0 *sig_cr_sec # incident photon density (3D), at a given transv. loc-n
        # eta = n0_phot *c_light *tau_pulse
        # gamma_degen = 1.0
        # en_gain = np.log( 1. +np.exp(sig_cr_sec *pop_inv *element.length) *(
        #             np.exp(gamma_degen *sig_cr_sec *eta) -1.0) ) /(gamma_degen *sig_cr_sec *eta)
        # return laser_pulse
        raise NotImplementedError(
            f'{self}.propagate() with prop_type="attenuate" is not currently supported'
        )

    def _propagate_placeholder(self, laser_pulse, calc_gain):
        # nslices = len(laser_pulse.slice)
        # for i in np.arange(nslices):
        #     print ('Pulse slice ', i+1, ' of ', nslices, ' propagated through crystal slice.')
        # return laser_pulse
        raise NotImplementedError(
            f'{self}.propagate() with prop_type="placeholder" is not currently supported'
        )

    def _propagate_n0n2_lct(self, laser_pulse, calc_gain):
        # print('prop_type = n0n2_lct')
        nslices_pulse = len(laser_pulse.slice)
        L_cryst = self.length
        n0 = self.n0
        n2 = self.n2
        # print('n0: %g, n2: %g' %(n0, n2))
        l_scale = self.l_scale

        photon_e_ev = laser_pulse.photon_e_ev

        ##Convert energy to wavelength
        hc_ev_um = 1.23984198  # hc [eV*um]
        phLambda = (
            hc_ev_um / photon_e_ev * 1e-6
        )  # wavelength corresponding to photon_e_ev in meters
        # print("Wavelength corresponding to %g keV: %g microns" %(photon_e_ev * 1e-3, phLambda / 1e-6))

        # calculate components of ABCD matrix corrected with wavelength and scale factor for use in LCT algorithm
        gamma = np.sqrt(n2 / n0)
        A = np.cos(gamma * L_cryst)
        B = (1 / gamma) * np.sin(gamma * L_cryst) * phLambda / (l_scale**2)
        C = -gamma * np.sin(gamma * L_cryst) / phLambda * (l_scale**2)
        D = np.cos(gamma * L_cryst)
        abcd_mat_cryst = np.array([[A, B], [C, D]])
        # print('A: %g' %A)
        # print('B: %g' %B)
        # print('C: %g' %C)
        # print('D: %g' %D)

        for i in np.arange(nslices_pulse):
            # i = 0
            thisSlice = laser_pulse.slice[i]
            if calc_gain:
                thisSlice = self.calc_gain(thisSlice)

            # construct 2d numpy complex E_field from pulse wfr object
            # pol = 6 in calc_int_from_wfr() for full electric
            # field (0 corresponds to horizontal, 1 corresponds to vertical polarization)
            wfr0 = thisSlice.wfr

            # horizontal component of electric field
            re0_ex, re0_mesh_ex = srwutil.calc_int_from_wfr(
                wfr0, _pol=0, _int_type=5, _det=None, _fname="", _pr=False
            )
            im0_ex, im0_mesh_ex = srwutil.calc_int_from_wfr(
                wfr0, _pol=0, _int_type=6, _det=None, _fname="", _pr=False
            )
            re0_2d_ex = np.array(re0_ex).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )
            im0_2d_ex = np.array(im0_ex).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )

            # vertical componenent of electric field
            re0_ey, re0_mesh_ey = srwutil.calc_int_from_wfr(
                wfr0, _pol=1, _int_type=5, _det=None, _fname="", _pr=False
            )
            im0_ey, im0_mesh_ey = srwutil.calc_int_from_wfr(
                wfr0, _pol=1, _int_type=6, _det=None, _fname="", _pr=False
            )
            re0_2d_ey = np.array(re0_ey).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )
            im0_2d_ey = np.array(im0_ey).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )

            Etot0_2d_x = re0_2d_ex + 1j * im0_2d_ex
            Etot0_2d_y = re0_2d_ey + 1j * im0_2d_ey

            xvals_slice = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            yvals_slice = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)

            dX = xvals_slice[1] - xvals_slice[0]  # horizontal spacing [m]
            dX_scale = dX / l_scale
            dY = yvals_slice[1] - yvals_slice[0]  # vertical spacing [m]
            dY_scale = dY / l_scale

            # define horizontal and vertical input signals
            in_signal_2d_x = (dX_scale, dY_scale, Etot0_2d_x)
            in_signal_2d_y = (dX_scale, dY_scale, Etot0_2d_y)

            # calculate 2D LCTs
            dX_out, dY_out, out_signal_2d_x = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_x
            )
            dX_out, dY_out, out_signal_2d_y = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_y
            )

            # extract propagated complex field and calculate corresponding x and y mesh arrays
            # we assume same mesh for both components of E_field
            hx = dX_out * l_scale
            hy = dY_out * l_scale
            # sig_arr_x = out_signal_2d_x
            # sig_arr_y = out_signal_2d_y
            ny, nx = np.shape(out_signal_2d_x)
            local_xv = rslct.lct_abscissae(nx, hx)
            local_yv = rslct.lct_abscissae(ny, hy)
            x_min = np.min(local_xv)
            x_max = np.max(local_xv)
            y_min = np.min(local_xv)
            y_max = np.max(local_xv)

            # return to SRW wavefront form
            ex_real = np.real(out_signal_2d_x).flatten(order="C")
            ex_imag = np.imag(out_signal_2d_x).flatten(order="C")

            ey_real = np.real(out_signal_2d_y).flatten(order="C")
            ey_imag = np.imag(out_signal_2d_y).flatten(order="C")

            ex_numpy = np.zeros(2 * len(ex_real))
            for i in range(len(ex_real)):
                ex_numpy[2 * i] = ex_real[i]
                ex_numpy[2 * i + 1] = ex_imag[i]

            ey_numpy = np.zeros(2 * len(ey_real))
            for i in range(len(ey_real)):
                ey_numpy[2 * i] = ey_real[i]
                ey_numpy[2 * i + 1] = ey_imag[i]

            ex = array.array("f", ex_numpy.tolist())
            ey = array.array("f", ey_numpy.tolist())

            wfr1 = srwlib.SRWLWfr(
                _arEx=ex,
                _arEy=ey,
                _typeE="f",
                _eStart=photon_e_ev,
                _eFin=photon_e_ev,
                _ne=1,
                _xStart=x_min,
                _xFin=x_max,
                _nx=nx,
                _yStart=y_min,
                _yFin=y_max,
                _ny=ny,
                _zStart=0.0,
                _partBeam=None,
            )

            thisSlice.wfr = wfr1

        # return wfr1
        return laser_pulse

    def _propagate_abcd_lct(self, laser_pulse, calc_gain):
        # print('prop_type = abcd_lct')
        nslices_pulse = len(laser_pulse.slice)
        l_scale = self.l_scale

        photon_e_ev = laser_pulse.photon_e_ev

        ##Convert energy to wavelength
        hc_ev_um = 1.23984198  # hc [eV*um]
        phLambda = (
            hc_ev_um / photon_e_ev * 1e-6
        )  # wavelength corresponding to photon_e_ev in meters
        # print("Wavelength corresponding to %g keV: %g microns" %(photon_e_ev * 1e-3, phLambda / 1e-6))

        # rescale ABCD matrix with wavelength and scale factor for use in LCT algorithm
        A = self.A
        B = self.B * phLambda / (l_scale**2)
        C = self.C / phLambda * (l_scale**2)
        D = self.D
        abcd_mat_cryst = np.array([[A, B], [C, D]])
        # print('A: %g' %A)
        # print('B: %g' %B)
        # print('C: %g' %C)
        # print('D: %g' %D)

        for i in np.arange(nslices_pulse):
            # i = 0
            thisSlice = laser_pulse.slice[i]
            if calc_gain:
                thisSlice = self.calc_gain(thisSlice)

            # construct 2d numpy complex E_field from pulse wfr object
            # pol = 6 in calc_int_from_wfr() for full electric
            # field (0 corresponds to horizontal, 1 corresponds to vertical polarization)
            wfr0 = thisSlice.wfr

            # horizontal component of electric field
            re0_ex, re0_mesh_ex = srwutil.calc_int_from_wfr(
                wfr0, _pol=0, _int_type=5, _det=None, _fname="", _pr=False
            )
            im0_ex, im0_mesh_ex = srwutil.calc_int_from_wfr(
                wfr0, _pol=0, _int_type=6, _det=None, _fname="", _pr=False
            )
            re0_2d_ex = np.array(re0_ex).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )
            im0_2d_ex = np.array(im0_ex).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )

            # vertical componenent of electric field
            re0_ey, re0_mesh_ey = srwutil.calc_int_from_wfr(
                wfr0, _pol=1, _int_type=5, _det=None, _fname="", _pr=False
            )
            im0_ey, im0_mesh_ey = srwutil.calc_int_from_wfr(
                wfr0, _pol=1, _int_type=6, _det=None, _fname="", _pr=False
            )
            re0_2d_ey = np.array(re0_ey).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )
            im0_2d_ey = np.array(im0_ey).reshape(
                (wfr0.mesh.nx, wfr0.mesh.ny), order="C"
            )

            Etot0_2d_x = re0_2d_ex + 1j * im0_2d_ex
            Etot0_2d_y = re0_2d_ey + 1j * im0_2d_ey

            xvals_slice = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            yvals_slice = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)

            dX = xvals_slice[1] - xvals_slice[0]  # horizontal spacing [m]
            dX_scale = dX / l_scale
            dY = yvals_slice[1] - yvals_slice[0]  # vertical spacing [m]
            dY_scale = dY / l_scale

            # define horizontal and vertical input signals
            in_signal_2d_x = (dX_scale, dY_scale, Etot0_2d_x)
            in_signal_2d_y = (dX_scale, dY_scale, Etot0_2d_y)

            # calculate 2D LCTs
            dX_out, dY_out, out_signal_2d_x = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_x
            )
            dX_out, dY_out, out_signal_2d_y = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_y
            )

            # extract propagated complex field and calculate corresponding x and y mesh arrays
            # we assume same mesh for both components of E_field
            hx = dX_out * l_scale
            hy = dY_out * l_scale
            # sig_arr_x = out_signal_2d_x
            # sig_arr_y = out_signal_2d_y
            ny, nx = np.shape(out_signal_2d_x)
            local_xv = rslct.lct_abscissae(nx, hx)
            local_yv = rslct.lct_abscissae(ny, hy)
            x_min = np.min(local_xv)
            x_max = np.max(local_xv)
            y_min = np.min(local_xv)
            y_max = np.max(local_xv)

            # return to SRW wavefront form
            ex_real = np.real(out_signal_2d_x).flatten(order="C")
            ex_imag = np.imag(out_signal_2d_x).flatten(order="C")

            ey_real = np.real(out_signal_2d_y).flatten(order="C")
            ey_imag = np.imag(out_signal_2d_y).flatten(order="C")

            ex_numpy = np.zeros(2 * len(ex_real))
            for i in range(len(ex_real)):
                ex_numpy[2 * i] = ex_real[i]
                ex_numpy[2 * i + 1] = ex_imag[i]

            ey_numpy = np.zeros(2 * len(ey_real))
            for i in range(len(ey_real)):
                ey_numpy[2 * i] = ey_real[i]
                ey_numpy[2 * i + 1] = ey_imag[i]

            ex = array.array("f", ex_numpy.tolist())
            ey = array.array("f", ey_numpy.tolist())

            wfr1 = srwlib.SRWLWfr(
                _arEx=ex,
                _arEy=ey,
                _typeE="f",
                _eStart=photon_e_ev,
                _eFin=photon_e_ev,
                _ne=1,
                _xStart=x_min,
                _xFin=x_max,
                _nx=nx,
                _yStart=y_min,
                _yFin=y_max,
                _ny=ny,
                _zStart=0.0,
                _partBeam=None,
            )

            thisSlice.wfr = wfr1

        # return wfr1
        return laser_pulse

    def _propagate_n0n2_srw(self, laser_pulse, calc_gain):
        # print('prop_type = n0n2_srw')
        nslices = len(laser_pulse.slice)
        L_cryst = self.length
        n0 = self.n0
        n2 = self.n2
        # print('n0: %g, n2: %g' %(n0, n2))

        for i in np.arange(nslices):
            thisSlice = laser_pulse.slice[i]
            if calc_gain:
                thisSlice = self.calc_gain(thisSlice)

            if n2 == 0:
                # print('n2 = 0')
                # A = 1.0
                # B = L_cryst
                # C = 0.0
                # D = 1.0
                optDrift = srwlib.SRWLOptD(L_cryst / n0)
                propagParDrift = [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
                # propagParDrift = [0, 0, 1., 0, 0, 1.1, 1.2, 1.1, 1.2, 0, 0, 0]
                optBL = srwlib.SRWLOptC([optDrift], [propagParDrift])
                # print("L_cryst/n0=",L_cryst/n0)
            else:
                # print('n2 .ne. 0')
                gamma = np.sqrt(n2 / n0)
                A = np.cos(gamma * L_cryst)
                B = (1 / gamma) * np.sin(gamma * L_cryst)
                C = -gamma * np.sin(gamma * L_cryst)
                D = np.cos(gamma * L_cryst)
                f1 = B / (1 - A)
                L = B
                f2 = B / (1 - D)

                optLens1 = srwlib.SRWLOptL(f1, f1)
                optDrift = srwlib.SRWLOptD(L)
                optLens2 = srwlib.SRWLOptL(f2, f2)

                propagParLens1 = [0, 0, 1.0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
                propagParDrift = [0, 0, 1.0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
                propagParLens2 = [0, 0, 1.0, 0, 0, 1, 1, 1, 1, 0, 0, 0]

                optBL = srwlib.SRWLOptC(
                    [optLens1, optDrift, optLens2],
                    [propagParLens1, propagParDrift, propagParLens2],
                )
                # optBL = createABCDbeamline(A,B,C,D)

            srwlib.srwl.PropagElecField(
                thisSlice.wfr, optBL
            )  # thisSlice s.b. a pointer, not a copy
            # print('Propagated pulse slice ', i+1, ' of ', nslices)
        return laser_pulse

    def _propagate_gain_calc(self, laser_pulse, calc_gain):
        # calculates gain regardles of calc_gain param value
        for i in np.arange(len(laser_pulse.slice)):
            thisSlice = laser_pulse.slice[i]
            thisSlice = self.calc_gain(thisSlice)
        return laser_pulse

    def propagate(self, laser_pulse, prop_type, calc_gain=False):
        return PKDict(
            attenuate=self._propagate_attenuate,
            placeholder=self._propagate_placeholder,
            abcd_lct=self._propagate_abcd_lct,
            n0n2_lct=self._propagate_n0n2_lct,
            n0n2_srw=self._propagate_n0n2_srw,
            gain_calc=self._propagate_gain_calc,
            default=super().propagate,
        )[prop_type](laser_pulse, calc_gain)

    def _interpolate_a_to_b(self, a, b):
        if a == "pop_inversion":
            # interpolate copy of pop_inversion to match lp_wfr
            temp_array = np.copy(self.pop_inversion_mesh)

            a_x = np.linspace(
                -self.population_inversion.mesh_extent,
                self.population_inversion.mesh_extent,
                self.population_inversion.n_cells,
            )
            a_y = a_x
            b_x = np.linspace(b.mesh.xStart, b.mesh.xFin, b.mesh.nx)
            b_y = np.linspace(b.mesh.yStart, b.mesh.yFin, b.mesh.ny)

        elif b == "pop_inversion":
            # interpolate copy of change_pop_inversion to match pop_inversion
            temp_array = np.copy(a.mesh)

            a_x = a.x
            a_y = a.y
            b_x = np.linspace(
                -self.population_inversion.mesh_extent,
                self.population_inversion.mesh_extent,
                self.population_inversion.n_cells,
            )
            b_y = b_x

        if not (np.array_equal(a_x, b_x) and np.array_equal(a_y, b_y)):

            # Create the spline for interpolation
            rect_biv_spline = RectBivariateSpline(a_x, a_y, temp_array)

            # Evaluate the spline at b gridpoints
            temp_array = rect_biv_spline(b_x, b_y)

            # Set any interpolated values outside the bounds of the original mesh to zero
            temp_array[b_x > np.max(a_x), :] = 0.0
            temp_array[b_x < np.min(a_x), :] = 0.0
            temp_array[:, b_y > np.max(a_y)] = 0.0
            temp_array[:, b_y < np.min(a_y)] = 0.0

        return temp_array

    def calc_gain(self, thisSlice):

        lp_wfr = thisSlice.wfr

        # Interpolate the excited state density mesh of the current crystal slice to
        # match the laser_pulse wavefront mesh
        temp_pop_inversion = self._interpolate_a_to_b("pop_inversion", lp_wfr)

        # Calculate gain
        absorp_cross_sec = 3.0e-23  # [m^2] 4.1e-23  #
        degen_factor = 1.67

        dx = (lp_wfr.mesh.xFin - lp_wfr.mesh.xStart) / lp_wfr.mesh.nx  # [m]
        dy = (lp_wfr.mesh.yFin - lp_wfr.mesh.yStart) / lp_wfr.mesh.ny  # [m]
        n_incident_photons = thisSlice.n_photons_2d.mesh / (dx * dy)  # [1/m^2]

        energy_gain = np.zeros(np.shape(n_incident_photons))
        gain_condition = np.where(n_incident_photons > 0)
        energy_gain[gain_condition] = (
            1.0 / (degen_factor * absorp_cross_sec * n_incident_photons[gain_condition])
        ) * np.log(
            1
            + np.exp(
                absorp_cross_sec * temp_pop_inversion[gain_condition] * self.length
            )
            * (
                np.exp(
                    degen_factor * absorp_cross_sec * n_incident_photons[gain_condition]
                )
                - 1.0
            )
        )

        # Calculate change factor for pop_inversion, note it has the same dimensions as lp_wfr
        change_pop_mesh = -(
            degen_factor * n_incident_photons * (energy_gain - 1.0) / self.length
        )
        change_pop_inversion = PKDict(
            mesh=change_pop_mesh,
            x=np.linspace(lp_wfr.mesh.xStart, lp_wfr.mesh.xFin, lp_wfr.mesh.nx),
            y=np.linspace(lp_wfr.mesh.yStart, lp_wfr.mesh.yFin, lp_wfr.mesh.ny),
        )

        # Interpolate the change to the excited state density mesh of the current crystal slice (change_pop_inversion)
        # so that it matches self.pop_inversion
        change_pop_inversion.mesh = self._interpolate_a_to_b(
            change_pop_inversion, "pop_inversion"
        )

        # Update the pop_inversion_mesh
        self.pop_inversion_mesh += change_pop_inversion.mesh

        # Update the number of photons
        thisSlice.n_photons_2d.mesh *= energy_gain

        # Update the wavefront itself: (KW To Do: make this a separate method?)
        #    First extract the electric fields

        # horizontal component of electric field
        re0_ex, re0_mesh_ex = srwutil.calc_int_from_wfr(
            lp_wfr, _pol=0, _int_type=5, _det=None, _fname="", _pr=False
        )
        im0_ex, im0_mesh_ex = srwutil.calc_int_from_wfr(
            lp_wfr, _pol=0, _int_type=6, _det=None, _fname="", _pr=False
        )
        gain_re0_ex = np.float64(re0_ex) * np.sqrt(energy_gain).flatten(order="C")
        gain_im0_ex = np.float64(im0_ex) * np.sqrt(energy_gain).flatten(order="C")

        # vertical componenent of electric field
        re0_ey, re0_mesh_ey = srwutil.calc_int_from_wfr(
            lp_wfr, _pol=1, _int_type=5, _det=None, _fname="", _pr=False
        )
        im0_ey, im0_mesh_ey = srwutil.calc_int_from_wfr(
            lp_wfr, _pol=1, _int_type=6, _det=None, _fname="", _pr=False
        )
        gain_re0_ey = np.float64(re0_ey) * np.sqrt(energy_gain).flatten(order="C")
        gain_im0_ey = np.float64(im0_ey) * np.sqrt(energy_gain).flatten(order="C")

        ex_numpy = np.zeros(2 * len(gain_re0_ex))
        for i in range(len(gain_re0_ex)):
            ex_numpy[2 * i] = gain_re0_ex[i]
            ex_numpy[2 * i + 1] = gain_im0_ex[i]

        ey_numpy = np.zeros(2 * len(gain_re0_ey))
        for i in range(len(gain_re0_ey)):
            ey_numpy[2 * i] = gain_re0_ey[i]
            ey_numpy[2 * i + 1] = gain_im0_ey[i]

        ex = array.array("f", ex_numpy.tolist())
        ey = array.array("f", ey_numpy.tolist())

        #    Pass changes to SRW
        wfr1 = srwlib.SRWLWfr(
            _arEx=ex,
            _arEy=ey,
            _typeE="f",
            _eStart=thisSlice.photon_e_ev,
            _eFin=thisSlice.photon_e_ev,
            _ne=1,
            _xStart=lp_wfr.mesh.xStart,
            _xFin=lp_wfr.mesh.xFin,
            _nx=lp_wfr.mesh.nx,
            _yStart=lp_wfr.mesh.yStart,
            _yFin=lp_wfr.mesh.yFin,
            _ny=lp_wfr.mesh.ny,
            _zStart=0.0,
            _partBeam=None,
        )

        thisSlice.wfr = wfr1
        return thisSlice
