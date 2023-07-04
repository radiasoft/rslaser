# -*- coding: utf-8 -*-
"""Definition of a crystal
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import numpy as np
import array
import math
import copy
from pykern.pkcollections import PKDict
from pykern.pkdebug import pkdp
import srwlib
from srwlib import srwl
import scipy.constants as const
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit
from scipy.special import gamma
from scipy.special import exp1
from rsmath import lct as rslct
from rslaser.utils.validator import ValidatorBase
from rslaser.utils import srwl_uti_data as srwutil
from rslaser.optics.element import ElementException, Element
from rslaser.thermal import ThermoOptic
from fenics import *
from mshr import *

_N_SLICE_DEFAULT = 50
_N0_DEFAULT = 1.75
_N2_DEFAULT = 0.001
_CRYSTAL_DEFAULTS = PKDict(
    n0=[_N0_DEFAULT for _ in range(_N_SLICE_DEFAULT)],
    n2=[_N2_DEFAULT for _ in range(_N_SLICE_DEFAULT)],
    delta_n_array=None,
    delta_n=None,
    delta_n_mesh_extent=0.01,  # range [m] of delta_n mesh assuming azimuthal symmetry
    length=0.2,  # [m], crystal length
    radius=0.01,  # [m], crystal radius
    l_scale=1,
    nslice=_N_SLICE_DEFAULT,
    slice_index=0,
    A=0.99765495,  # A = 9.99988571e-01,
    B=1.41975385,  # B = 1.99999238e-01,
    C=-0.0023775,  # C = -1.14285279e-04,
    D=0.99896716,  # D = 9.99988571e-01,
    radial_n2_factor=1.3,
    alpha=120.0,  # [1/m], absorption coefficient for Al2O3
    rho=3980.0,  # [kg/m^3], density for Al2O3
    Kc=33.0,  # [W/m K], thermal conductivity for Al203
    cp=756.0,  # [J/kg K], specific heat capacity (constant pressure) for Al203
    Tc=0.0,  # [C], coolant (or ambient) temperature outside the crystal
    lambda_seed=800.0,  # [nm], seed wavelength for thermo-optic coupling factor
    lambda_pump=532.0,  # [nm], pump laser operating wavelength
    pump_waist=0.00164,  # [m] pump laser waist
    pump_energy=0.0211,  # [J], pump laser energy onto the crystal
    pump_type="dual",  # pumping type, one of: "left", "right", or "dual"
    pump_profile="hog",  # pump profile, one of: "uniform", "gaussian", "hog", or "tophat"
    pump_gorder=2.0,  # Gaussian order of the pump profile, used for "hog" profile only
    pump_offset_x=0.0,  # [m], horizontal offset of pump laser center from crystal center
    pump_offset_y=0.0,  # [m], veritcal offset of pump laser center from crystal center
    pump_rate=1.0e3,  # [s^-1], repetition rate of the pump laser
    num_cells=64,  # number of cells used in propagation calculations
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
            l_scale: length scale factor for L3CT propagation
    """

    _DEFAULTS = _CRYSTAL_DEFAULTS
    _INPUT_ERROR = ElementException

    def __init__(self, params=None):

        # Validate & set input parameters
        params = self._get_params(params)
        self._validate_params(params)
        for p, param in params.items():
            setattr(self, p, param)

        # Check if n2<0, throw an exception if true
        if (np.array(params.n2) < 0.0).any():
            raise self._INPUT_ERROR(f"You've specified negative value(s) for n2")
        self.slice = []

        for j in range(self.nslice):
            p = params.copy()
            p.update(
                PKDict(
                    n0=params.n0[j],
                    n2=params.n2[j],
                    delta_n=params.delta_n_array[j]
                    if params.delta_n_array is not None
                    else None,
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

    def propagate(
        self, laser_pulse, prop_type, calc_gain=False, radial_n2=False, nl_kick=False
    ):
        assert (laser_pulse.pulse_direction == 0.0) or (
            laser_pulse.pulse_direction == 180.0
        ), "ERROR -- Propagation not implemented for the pulse direction {}".format(
            laser_pulse.pulse_direction
        )

        if laser_pulse.pulse_direction == 0.0:
            slice_array = self.slice
        elif laser_pulse.pulse_direction == 180.0:
            slice_array = self.slice[::-1]

        # Iterate through laser_pulse and offset all of the fields
        if self.slice[0].pump_offset_x != 0 or self.slice[0].pump_offset_y != 0:
            laser_pulse.shift_wavefront(
                self.slice[0].pump_offset_x,
                self.slice[0].pump_offset_y,
            )

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

                laser_pulse = laser_pulse.combine_n2_variation(
                    laser_pulse_copies,
                    s.radial_n2_factor,
                    s.pump_waist,
                    s.n2,
                )
            else:
                laser_pulse = s.propagate(laser_pulse, prop_type, calc_gain, nl_kick)
            # laser_pulse.resize_laser_mesh()
            # laser_pulse.flatten_phase_edges()

        # Iterate through laser_pulse and return all of the fields
        if self.slice[0].pump_offset_x != 0 or self.slice[0].pump_offset_y != 0:
            laser_pulse.shift_wavefront(
                -self.slice[0].pump_offset_x,
                -self.slice[0].pump_offset_y,
            )
        laser_pulse.resize_laser_mesh()
        laser_pulse.flatten_phase_edges()
        return laser_pulse

    def calc_n0n2(
        self, set_n=False, mesh_density=50, method="analytical", heat_load="gaussian"
    ):
        # mesh_density [int]: value ≥ 120 will produce more accurate results; slower, but closer to numerical conversion

        # Validate choice of solution method
        method = method.lower()
        if method not in ("fenics", "analytical"):
            raise ValueError("'method' must be either 'fenics' or 'analytical'")

        # Initialize a thermo-optic simulator object
        TO_Sim = ThermoOptic(self, mesh_density)

        # Set evaluation points for thermo-optic calculations
        n_radpts = 100  # no. of radial points at which to extract data
        n_longpts = self.nslice  # no. of longitudinal points at which to extract data
        TO_Sim.set_points((n_radpts, 0, n_longpts))

        # For high rep-rates, solve steady-state heat equation
        if method == "fenics":

            # Set boundary values for thermo-optic simulations
            bc_tol = 2.0 * self.radius * (self.radius / 40.0)  # 2 * rad * delta(rad)
            TO_Sim.set_boundary(bc_tol)

            # Set thermal load & carry out thermo-optic simulation
            TO_Sim.set_load(heat_load)
            Trz = TO_Sim.solve_steady()

        # For analytical solutions, compute Innocenzi solution
        elif method == "analytical":
            TO_Sim.set_load(heat_load)
            Trz = getattr(TO_Sim, heat_load + "_solution")()

        # Compute indices of refraction & ABCD matrices for each slice
        nT, nFit = TO_Sim.compute_indices(Trz)
        ABCDs, full_ABCD = TO_Sim.compute_ABCD(nFit)

        # Set n0/n2 values for crystal slices if desired
        if set_n:
            for s in self.slice:
                s.n0 = nFit[s.slice_index, 0, 0]
                s.n2 = nFit[s.slice_index, 0, 1]

        return nFit[:, 0, 0], nFit[:, 0, 1], full_ABCD

    def extract_excited_states(self):
        long_excited_states = np.zeros(self.nslice)
        trans_excited_states = np.zeros((self.params.num_cells, self.params.num_cells))
        for j in range(self.nslice):
            thisSlice = self.slice[j]
            dx = (2.0 * thisSlice.radius) / thisSlice.num_cells
            cell_area = dx**2.0 * thisSlice.length
            trans_excited_states += thisSlice.pop_inversion_mesh * cell_area
            long_excited_states[j] = np.sum(thisSlice.pop_inversion_mesh * cell_area)

        return long_excited_states, trans_excited_states


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

        # Validate & set input parameters
        params = self._get_params(params)
        self._validate_params(params)
        for p, param in params.items():
            setattr(self, p, param)
        self.prop_type = "srw"  # Default prop_type for element.py propagation

        # Wavelength-dependent cross-section (P. F. Moulton, 1986)
        wavelength = np.array(
            [600, 625, 650, 700, 750, 800, 850, 900, 950, 1000, 1025, 1050]
        ) * (1.0e-9)
        cross_section = np.array(
            [
                0.0,
                0.02,
                0.075,
                0.437,
                0.845,
                0.99,
                0.815,
                0.6,
                0.415,
                0.276,
                0.255,
                0.247,
            ]
        ) * (4.8e-23)
        self.cross_section_fn = splrep(wavelength, cross_section)

        #  Assuming wfr0 exsts, created e.g. via
        #  wfr0=createGsnSrcSRW(sigrW,propLen,pulseE,poltype,photon_e_ev,sampFact,mx,my)
        # n_x = wfr0.mesh.nx  #  nr of grid points in x
        # n_y = wfr0.mesh.ny  #  nr of grid points in y
        # sig_cr_sec = np.ones((n_x, n_y), dtype=np.float32)

        # create mesh for delta_n array
        self.delta_n_xstart = -params.delta_n_mesh_extent
        self.delta_n_xfin = params.delta_n_mesh_extent

        # 2d mesh of excited state density (sigma)
        self._initialize_excited_states_mesh(params, params.nslice)

    def _left_pump(self, nslice, xv, yv):

        # z = distance from left of crystal to center of current slice (assumes all crystal slices have same length)
        z = self.length * (self.slice_index + 0.5)

        slice_front = z - (self.length / 2.0)
        slice_end = z + (self.length / 2.0)

        # calculate correction factor for representing a gaussian pulse with a series of flat-top slices
        correction_factor = (
            (np.exp(-self.alpha * slice_front) - np.exp(-self.alpha * slice_end))
            / self.alpha
        ) / (np.exp(-self.alpha * z) * self.length)

        # integrate super-gaussian
        integral_factor = (
            2 ** ((self.pump_gorder - 2.0) / self.pump_gorder)
            * gamma(2 / self.pump_gorder)
        ) / (
            self.pump_gorder
            * (1 / (self.pump_waist**self.pump_gorder)) ** (2.0 / self.pump_gorder)
        )

        pump_wavelength = 532.0  # [nm]
        seed_wavelength = 800.0  # [nm]
        fraction_to_heating = (seed_wavelength - pump_wavelength) / seed_wavelength

        # Create a default mesh of [num_excited_states/m^3]
        pop_inversion_mesh = (
            (self.lambda_pump / (const.h * const.c))
            * (
                (
                    (1 - np.exp(-self.alpha * self.length * nslice))
                    * (1.0 - fraction_to_heating)
                    * self.pump_energy
                    * np.exp(
                        -2.0
                        * (
                            np.sqrt(
                                (xv - self.pump_offset_x) ** 2.0
                                + (yv - self.pump_offset_y) ** 2.0
                            )
                            / self.pump_waist
                        )
                        ** self.pump_gorder
                    )
                )
                / (const.pi * integral_factor)
            )
            * np.exp(-self.alpha * z)
            * correction_factor
        ) / (self.length * nslice)
        left_tuple = (z, slice_front, slice_end)

        return np.array((left_tuple,))

    def _right_pump(self, nslice, xv, yv):

        # z = distance from right of crystal to center of current slice (assumes all crystal slices have same length)
        z = self.length * ((nslice - self.slice_index - 1) + 0.5)

        slice_front = z - (self.length / 2.0)
        slice_end = z + (self.length / 2.0)

        # calculate correction factor for representing a gaussian pulse with a series of flat-top slices
        correction_factor = (
            (np.exp(-self.alpha * slice_front) - np.exp(-self.alpha * slice_end))
            / self.alpha
        ) / (np.exp(-self.alpha * z) * self.length)

        # integrate super-gaussian
        integral_factor = (
            2 ** ((self.pump_gorder - 2.0) / self.pump_gorder)
            * gamma(2 / self.pump_gorder)
        ) / (
            self.pump_gorder
            * (1 / (self.pump_waist**self.pump_gorder)) ** (2.0 / self.pump_gorder)
        )

        # Create a default mesh of [num_excited_states/m^3]
        fraction_to_heating = (self.lambda_seed - self.lambda_pump) / self.lambda_seed
        pop_inversion_mesh = (
            (self.lambda_pump / (const.h * const.c))
            * (
                (
                    (1 - np.exp(-self.alpha * self.length * nslice))
                    * (1.0 - fraction_to_heating)
                    * self.pump_energy
                    * np.exp(
                        -2.0
                        * (
                            np.sqrt(
                                (xv - self.pump_offset_x) ** 2.0
                                + (yv - self.pump_offset_y) ** 2.0
                            )
                            / self.pump_waist
                        )
                        ** self.pump_gorder
                    )
                )
                / (const.pi * integral_factor)
            )
            * np.exp(-self.alpha * z)
            * correction_factor
        ) / (self.length * nslice)
        right_tuple = (z, slice_front, slice_end)

        return np.array((right_tuple,))

    def _dual_pump(self, nslice, xv, yv):
        left_tuple = self._left_pump(nslice, xv, yv)
        right_tuple = self._right_pump(nslice, xv, yv)
        return np.concatenate((left_tuple, right_tuple))

    def _initialize_excited_states_mesh(self, params, nslice):
        x = np.linspace(
            -self.radius,
            self.radius,
            self.num_cells,
        )
        xv, yv = np.meshgrid(x, x)

        param_set_array = PKDict(
            dual=self._dual_pump,
            left=self._left_pump,
            right=self._right_pump,
        )[self.pump_type](nslice, xv, yv)

        pop_inversion_mesh = np.zeros((len(x), len(x)))
        for param_set in param_set_array:
            z = param_set[0]
            slice_front = param_set[1]
            slice_end = param_set[2]

            # calculate correction factor for representing a gaussian pulse with a series of flat-top slices
            correction_factor = (
                (np.exp(-self.alpha * slice_front) - np.exp(-self.alpha * slice_end))
                / self.alpha
            ) / (np.exp(-self.alpha * z) * self.length)

            # integrate super-gaussian
            integral_factor = (
                2 ** ((self.pump_gorder - 2.0) / self.pump_gorder)
                * gamma(2 / self.pump_gorder)
            ) / (
                self.pump_gorder
                * (1 / (self.pump_waist**self.pump_gorder))
                ** (2.0 / self.pump_gorder)
            )
            fraction_to_heating = 1.0 - self.lambda_pump / self.lambda_seed

            # Create mesh of [num_excited_states/m^3] pop_inversion_mesh
            temp_mesh = (
                (self.lambda_pump / (const.h * const.c))
                * (
                    (
                        (1.0 - np.exp(-self.alpha * self.length * nslice))
                        * (1.0 - fraction_to_heating)
                        * self.pump_energy
                        * np.exp(
                            -2.0
                            * (
                                np.sqrt(
                                    (xv - self.pump_offset_x) ** 2.0
                                    + (yv - self.pump_offset_y) ** 2.0
                                )
                                / self.pump_waist
                            )
                            ** self.pump_gorder
                        )
                    )
                    / (const.pi * integral_factor)
                )
                * np.exp(-self.alpha * z)
                * correction_factor
            ) / (self.length * nslice)
            pop_inversion_mesh += temp_mesh

        self.pop_inversion_mesh = pop_inversion_mesh

    def _propagate_attenuate(self, laser_pulse, calc_gain, nl_kick):
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

    def _propagate_placeholder(self, laser_pulse, calc_gain, nl_kick):
        # nslices = len(laser_pulse.slice)
        # for i in np.arange(nslices):
        #     print ('Pulse slice ', i+1, ' of ', nslices, ' propagated through crystal slice.')
        # return laser_pulse
        raise NotImplementedError(
            f'{self}.propagate() with prop_type="placeholder" is not currently supported'
        )

    def _propagate_n0n2_lct(self, laser_pulse, calc_gain, nl_kick):
        nslices_pulse = len(laser_pulse.slice)
        L_cryst = self.length
        n0 = self.n0
        n2 = self.n2
        l_scale = self.l_scale

        photon_e_ev = laser_pulse.photon_e_ev

        ##Convert energy to wavelength
        hc_ev_um = 1.23984198  # hc [eV*um]
        phLambda = (
            hc_ev_um / photon_e_ev * 1e-6
        )  # wavelength corresponding to photon_e_ev in meters

        # calculate components of ABCD matrix corrected with wavelength and scale factor for use in LCT algorithm
        gamma = np.sqrt(n2 / n0)
        A = np.cos(gamma * L_cryst)
        B = phLambda * L_cryst / (l_scale**2) * np.sinc(gamma * L_cryst / np.pi)
        C = -gamma * np.sin(gamma * L_cryst) / phLambda * (l_scale**2)
        D = np.cos(gamma * L_cryst)
        abcd_mat_cryst = np.array([[A, B], [C, D]])

        for i in np.arange(nslices_pulse):
            # i = 0
            thisSlice = laser_pulse.slice[i]
            if calc_gain:
                thisSlice = self.calc_gain(thisSlice)
            if nl_kick:
                thisSlice = self.nl_kick(thisSlice)

            wfr0 = thisSlice.wfr
            re0_2d_ex, im0_2d_ex, re0_2d_ey, im0_2d_ey = srwutil.extract_2d_fields(
                thisSlice.wfr
            )

            xvals_slice = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            yvals_slice = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)

            mesh_old = {
                "re0_2d_ex": re0_2d_ex,
                "im0_2d_ex": im0_2d_ex,
                "re0_2d_ey": re0_2d_ey,
                "im0_2d_ey": im0_2d_ey,
            }
            xvals_slice, yvals_slice, mesh_new = _interp_to_odd(
                xvals_slice, yvals_slice, mesh_old
            )

            Etot0_2d_x = mesh_new["re0_2d_ex"] + 1j * mesh_new["im0_2d_ex"]
            Etot0_2d_y = mesh_new["re0_2d_ey"] + 1j * mesh_new["im0_2d_ey"]

            dX = xvals_slice[1] - xvals_slice[0]  # horizontal spacing [m]
            dX_scale = dX / l_scale
            dY = yvals_slice[1] - yvals_slice[0]  # vertical spacing [m]
            dY_scale = dY / l_scale

            # define horizontal and vertical input signals
            in_signal_2d_x = (dX_scale, dY_scale, Etot0_2d_x)
            in_signal_2d_y = (dX_scale, dY_scale, Etot0_2d_y)

            assert np.shape(Etot0_2d_x)[0] % 2 != 0, "ERROR -- nx is even"
            assert np.shape(Etot0_2d_x)[1] % 2 != 0, "ERROR -- ny is even"

            # calculate 2D LCTs
            dX_out, dY_out, out_signal_2d_x = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_x
            )
            dX_out, dY_out, out_signal_2d_y = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_y
            )

            assert np.shape(out_signal_2d_x) == np.shape(
                out_signal_2d_y
            ), "ERROR -- x and y output not equal"

            re_out_signal_2d_x = np.real(out_signal_2d_x)
            x_total = (np.shape(re_out_signal_2d_x)[0] - 1) * dX_out
            y_total = (np.shape(re_out_signal_2d_x)[1] - 1) * dY_out
            xold = np.linspace(
                -x_total / 2.0, x_total / 2.0, np.shape(re_out_signal_2d_x)[0]
            )
            yold = np.linspace(
                -y_total / 2.0, y_total / 2.0, np.shape(re_out_signal_2d_x)[1]
            )

            mesh_old_2 = {
                "re_out_signal_2d_x": np.real(out_signal_2d_x),
                "im_out_signal_2d_x": np.imag(out_signal_2d_x),
                "re_out_signal_2d_y": np.real(out_signal_2d_y),
                "im_out_signal_2d_y": np.imag(out_signal_2d_y),
            }
            xnew, ynew, mesh_new = _interp_to_odd(xold, yold, mesh_old_2)

            if (
                np.shape(re_out_signal_2d_x)[0] % 2 == 0
                or np.shape(re_out_signal_2d_x)[1] % 2 == 0
            ):
                dX_out = np.mean(np.diff(xnew))
                dY_out = np.mean(np.diff(ynew))

            out_signal_2d_x = (
                mesh_new["re_out_signal_2d_x"] + 1j * mesh_new["im_out_signal_2d_x"]
            )
            out_signal_2d_y = (
                mesh_new["re_out_signal_2d_y"] + 1j * mesh_new["im_out_signal_2d_y"]
            )

            # extract propagated complex field and calculate corresponding x and y mesh arrays
            # we assume same mesh for both components of E_field
            hx = dX_out * l_scale
            hy = dY_out * l_scale
            ny, nx = np.shape(out_signal_2d_x)
            local_xv = rslct.lct_abscissae(nx, hx)
            local_yv = rslct.lct_abscissae(ny, hy)

            # return to SRW wavefront form
            re_ex = np.real(out_signal_2d_x)
            im_ex = np.imag(out_signal_2d_x)
            re_ey = np.real(out_signal_2d_y)
            im_ey = np.imag(out_signal_2d_y)

            x = np.linspace(np.min(local_xv), np.max(local_xv), nx)
            y = np.linspace(np.min(local_xv), np.max(local_xv), ny)

            # remake the wavefront
            thisSlice.wfr = srwutil.make_wavefront(
                re_ex,
                im_ex,
                re_ey,
                im_ey,
                thisSlice.photon_e_ev,
                x,
                y,
            )

        return laser_pulse

    def _propagate_abcd_lct(self, laser_pulse, calc_gain, nl_kick):
        nslices_pulse = len(laser_pulse.slice)
        l_scale = self.l_scale

        photon_e_ev = laser_pulse.photon_e_ev

        ##Convert energy to wavelength
        hc_ev_um = 1.23984198  # hc [eV*um]
        phLambda = (
            hc_ev_um / photon_e_ev * 1e-6
        )  # wavelength corresponding to photon_e_ev in meters

        # rescale ABCD matrix with wavelength and scale factor for use in LCT algorithm
        A = self.A
        B = self.B * phLambda / (l_scale**2)
        C = self.C / phLambda * (l_scale**2)
        D = self.D
        abcd_mat_cryst = np.array([[A, B], [C, D]])

        for i in np.arange(nslices_pulse):
            thisSlice = laser_pulse.slice[i]

            if calc_gain:
                thisSlice = self.calc_gain(thisSlice)
            if nl_kick:
                thisSlice = self.nl_kick(thisSlice)

            wfr0 = thisSlice.wfr
            re0_2d_ex, im0_2d_ex, re0_2d_ey, im0_2d_ey = srwutil.extract_2d_fields(
                thisSlice.wfr
            )

            xvals_slice = np.linspace(wfr0.mesh.xStart, wfr0.mesh.xFin, wfr0.mesh.nx)
            yvals_slice = np.linspace(wfr0.mesh.yStart, wfr0.mesh.yFin, wfr0.mesh.ny)

            mesh_old = {
                "re0_2d_ex": re0_2d_ex,
                "im0_2d_ex": im0_2d_ex,
                "re0_2d_ey": re0_2d_ey,
                "im0_2d_ey": im0_2d_ey,
            }
            xvals_slice, yvals_slice, mesh_new = _interp_to_odd(
                xvals_slice, yvals_slice, mesh_old
            )

            Etot0_2d_x = mesh_new["re0_2d_ex"] + 1j * mesh_new["im0_2d_ex"]
            Etot0_2d_y = mesh_new["re0_2d_ey"] + 1j * mesh_new["im0_2d_ey"]

            dX = xvals_slice[1] - xvals_slice[0]  # horizontal spacing [m]
            dX_scale = dX / l_scale
            dY = yvals_slice[1] - yvals_slice[0]  # vertical spacing [m]
            dY_scale = dY / l_scale

            # define horizontal and vertical input signals
            in_signal_2d_x = (dX_scale, dY_scale, Etot0_2d_x)
            in_signal_2d_y = (dX_scale, dY_scale, Etot0_2d_y)

            assert np.shape(Etot0_2d_x)[0] % 2 != 0, "ERROR -- nx is even"
            assert np.shape(Etot0_2d_x)[1] % 2 != 0, "ERROR -- ny is even"

            # calculate 2D LCTs
            dX_out, dY_out, out_signal_2d_x = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_x
            )
            dX_out, dY_out, out_signal_2d_y = rslct.apply_lct_2d_sep(
                abcd_mat_cryst, abcd_mat_cryst, in_signal_2d_y
            )

            assert np.shape(out_signal_2d_x) == np.shape(
                out_signal_2d_y
            ), "ERROR -- x and y output not equal"

            re_out_signal_2d_x = np.real(out_signal_2d_x)
            x_total = (np.shape(re_out_signal_2d_x)[0] - 1) * dX_out
            y_total = (np.shape(re_out_signal_2d_x)[1] - 1) * dY_out
            xold = np.linspace(
                -x_total / 2.0, x_total / 2.0, np.shape(re_out_signal_2d_x)[0]
            )
            yold = np.linspace(
                -y_total / 2.0, y_total / 2.0, np.shape(re_out_signal_2d_x)[1]
            )

            mesh_old_2 = {
                "re_out_signal_2d_x": np.real(out_signal_2d_x),
                "im_out_signal_2d_x": np.imag(out_signal_2d_x),
                "re_out_signal_2d_y": np.real(out_signal_2d_y),
                "im_out_signal_2d_y": np.imag(out_signal_2d_y),
            }
            xnew, ynew, mesh_new = _interp_to_odd(xold, yold, mesh_old_2)

            if (
                np.shape(re_out_signal_2d_x)[0] % 2 == 0
                or np.shape(re_out_signal_2d_x)[1] % 2 == 0
            ):
                dX_out = np.mean(np.diff(xnew))
                dY_out = np.mean(np.diff(ynew))

            out_signal_2d_x = (
                mesh_new["re_out_signal_2d_x"] + 1j * mesh_new["im_out_signal_2d_x"]
            )
            out_signal_2d_y = (
                mesh_new["re_out_signal_2d_y"] + 1j * mesh_new["im_out_signal_2d_y"]
            )

            # extract propagated complex field and calculate corresponding x and y mesh arrays
            # we assume same mesh for both components of E_field
            hx = dX_out * l_scale
            hy = dY_out * l_scale
            ny, nx = np.shape(out_signal_2d_x)

            assert np.shape(out_signal_2d_x)[0] % 2 != 0, "ERROR -- nx is even"
            assert np.shape(out_signal_2d_x)[1] % 2 != 0, "ERROR -- ny is even"

            local_xv = rslct.lct_abscissae(nx, hx)
            local_yv = rslct.lct_abscissae(ny, hy)

            # return to SRW wavefront form
            re_ex = np.real(out_signal_2d_x)
            im_ex = np.imag(out_signal_2d_x)
            re_ey = np.real(out_signal_2d_y)
            im_ey = np.imag(out_signal_2d_y)

            x = np.linspace(np.min(local_xv), np.max(local_xv), nx)
            y = np.linspace(np.min(local_xv), np.max(local_xv), ny)

            # remake the wavefront
            thisSlice.wfr = srwutil.make_wavefront(
                re_ex,
                im_ex,
                re_ey,
                im_ey,
                thisSlice.photon_e_ev,
                x,
                y,
            )

        return laser_pulse

    def _propagate_n0n2_srw(self, laser_pulse, calc_gain, nl_kick):
        nslices = len(laser_pulse.slice)
        L_cryst = self.length
        n0 = self.n0
        n2 = self.n2

        for i in np.arange(nslices):
            thisSlice = laser_pulse.slice[i]
            if calc_gain:
                thisSlice = self.calc_gain(thisSlice)
            if nl_kick:
                thisSlice = self.nl_kick(thisSlice)

            if n2 == 0:
                optDrift = srwlib.SRWLOptD(L_cryst / n0)
                propagParDrift = [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0]
                # propagParDrift = [0, 0, 1., 0, 0, 1.1, 1.2, 1.1, 1.2, 0, 0, 0]
                optBL = srwlib.SRWLOptC([optDrift], [propagParDrift])

            else:
                gamma = np.sqrt(n2 / n0)
                A = np.cos(gamma * L_cryst)
                B = L_cryst * np.sinc(gamma * L_cryst / np.pi)
                # B = (1 / gamma) * np.sin(gamma * L_cryst)
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

    def _propagate_gain_calc(self, laser_pulse, calc_gain, nl_kick):
        # calculates gain regardles of calc_gain param value
        for i in np.arange(len(laser_pulse.slice)):
            thisSlice = laser_pulse.slice[i]
            thisSlice = self.calc_gain(thisSlice)
        return laser_pulse

    def _propagate_nl_kick(self, laser_pulse, nl_kick):
        # applies NL kick regardless of nl_kick param value
        for i in np.arange(len(laser_pulse.slice)):
            thisSlice = laser_pulse.slice[i]
            thisSlice = self.nl_kick(thisSlice)
        return laser_pulse

    def propagate(self, laser_pulse, prop_type, calc_gain=False, nl_kick=False):
        if prop_type == "default":
            super().propagate(laser_pulse)
            return
        r = PKDict(
            attenuate=self._propagate_attenuate,
            placeholder=self._propagate_placeholder,
            abcd_lct=self._propagate_abcd_lct,
            n0n2_lct=self._propagate_n0n2_lct,
            n0n2_srw=self._propagate_n0n2_srw,
            gain_calc=self._propagate_gain_calc,
        )[prop_type](laser_pulse, calc_gain, nl_kick)
        return r

    def _interpolate_a_to_b(self, a, b):
        if a == "pop_inversion":
            # interpolate copy of pop_inversion to match lp_wfr
            temp_array = np.copy(self.pop_inversion_mesh)

            a_x = np.linspace(
                -self.radius,
                self.radius,
                self.num_cells,
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
                -self.radius,
                self.radius,
                self.num_cells,
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

    def delta_n_to_wfr_interp(self, delta_n_array, radpts, wfr_xvals, wfr_yvals):
        """
        This function interpolates an input delta_n array to match
        the shape and range of a wavefront array (wfr_xvals, wfr_yvals)
        if they are of different shape and/or range.

        delta_n_array: input delta n array
        radpts: delta_n array radius points [m] note: we assume azimuthal symmetry
        wfr_xvals: wfr array horizontal points [m]
        wfr_yvals: wfr array vertical points [m]
        """
        if np.shape(delta_n_array)[0] != np.size(wfr_xvals) or np.shape(delta_n_array)[
            1
        ] != np.size(wfr_yvals):

            # interpolate delta_n array to match shape of wfr mesh
            delta_n_interp_func = RegularGridInterpolator(
                (radpts, radpts), delta_n_array, method="linear", bounds_error=False
            )
            X, Y = np.meshgrid(wfr_xvals, wfr_yvals)
            delta_n_array_interp = delta_n_interp_func((X, Y))

        else:
            delta_n_array_interp = delta_n_array

        return delta_n_array_interp

    def calc_gain(self, thisSlice):

        lp_wfr = thisSlice.wfr

        # Interpolate the excited state density mesh of the current crystal slice to
        # match the laser_pulse wavefront mesh
        temp_pop_inversion = self._interpolate_a_to_b("pop_inversion", lp_wfr)

        # Calculate gain
        cross_sec = splev(thisSlice._lambda0, self.cross_section_fn)  # [m^2]
        degen_factor = 1.67

        dx = (lp_wfr.mesh.xFin - lp_wfr.mesh.xStart) / lp_wfr.mesh.nx  # [m]
        dy = (lp_wfr.mesh.yFin - lp_wfr.mesh.yStart) / lp_wfr.mesh.ny  # [m]
        n_incident_photons = thisSlice.n_photons_2d.mesh / (dx * dy)  # [1/m^2]

        energy_gain = np.zeros(np.shape(n_incident_photons))
        gain_condition = np.where(n_incident_photons > 0)
        energy_gain[gain_condition] = (
            1.0 / (degen_factor * cross_sec * n_incident_photons[gain_condition])
        ) * np.log(
            1
            + np.exp(cross_sec * temp_pop_inversion[gain_condition] * self.length)
            * (
                np.exp(degen_factor * cross_sec * n_incident_photons[gain_condition])
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

        # Update the wavefront itself
        """
        intensity_2d = srwutil.calc_int_from_elec(lp_wfr)
        phase_1d = srwlib.array("d", [0] * lp_wfr.mesh.nx * lp_wfr.mesh.ny)
        srwl.CalcIntFromElecField(phase_1d, lp_wfr, 0, 4, 3, lp_wfr.mesh.eStart, 0, 0)
        phase_2d = (
            np.array(phase_1d)
            .reshape((lp_wfr.mesh.nx, lp_wfr.mesh.ny), order="C")
            .astype(np.float64)
        )

        gain_intensity = intensity_2d * energy_gain
        gain_phase = phase_2d

        gain_e_norm = np.sqrt(2.0 * gain_intensity / (const.c * const.epsilon_0))
        gain_re0_ex = np.multiply(gain_e_norm, np.cos(gain_phase))
        gain_im0_ex = np.multiply(gain_e_norm, np.sin(gain_phase))
        gain_re0_ey = np.zeros(np.shape(gain_re0_ex))
        gain_im0_ey = np.zeros(np.shape(gain_im0_ex))
        """
        re0_2d_ex, im0_2d_ex, re0_2d_ey, im0_2d_ey = srwutil.extract_2d_fields(
            thisSlice.wfr
        )

        gain_re0_ex = re0_2d_ex * np.sqrt(energy_gain)
        gain_im0_ex = im0_2d_ex * np.sqrt(energy_gain)
        gain_re0_ey = re0_2d_ey * np.sqrt(energy_gain)
        gain_im0_ey = im0_2d_ey * np.sqrt(energy_gain)
        # """

        x = np.linspace(lp_wfr.mesh.xStart, lp_wfr.mesh.xFin, lp_wfr.mesh.nx)
        y = np.linspace(lp_wfr.mesh.yStart, lp_wfr.mesh.yFin, lp_wfr.mesh.ny)

        # remake the wavefront
        thisSlice.wfr = srwutil.make_wavefront(
            gain_re0_ex,
            gain_im0_ex,
            gain_re0_ey,
            gain_im0_ey,
            thisSlice.photon_e_ev,
            x,
            y,
        )

        return thisSlice

    def nl_kick(self, thisSlice):
        radpts = np.linspace(
            self.delta_n_xstart, self.delta_n_xfin, np.size(self.delta_n[0])
        )
        radpts_m = radpts / 1e2

        # calculate wavefront mesh values

        lp_wfr = thisSlice.wfr
        wfr_xvals = np.linspace(lp_wfr.mesh.xStart, lp_wfr.mesh.xFin, lp_wfr.mesh.nx)
        wfr_yvals = np.linspace(lp_wfr.mesh.yStart, lp_wfr.mesh.yFin, lp_wfr.mesh.ny)
        delta_n_interp = self.delta_n_to_wfr_interp(
            self.delta_n, radpts_m, wfr_xvals, wfr_yvals
        )

        # calculate wavelength [m]  from input energy
        hc_ev_um = 1.23984198  # hc [eV*um]
        phLambda = hc_ev_um / thisSlice.photon_e_ev * 1e-6
        l_over_lam = self.length / phLambda
        # print('l_over_lam: %g' %l_over_lam)

        # create nonlinear kick array
        nl_kick_array = np.exp(np.multiply(np.multiply(delta_n_interp, 1j), l_over_lam))

        wfr0 = thisSlice.wfr
        re0_2d_ex, im0_2d_ex, re0_2d_ey, im0_2d_ey = srwutil.extract_2d_fields(wfr0)

        Etot0_2d_x = re0_2d_ex + 1j * im0_2d_ex
        Etot0_2d_y = re0_2d_ey + 1j * im0_2d_ey

        # multiply horizontal and vertical total E fields by nl kick array
        Etot0_2d_x_nl_kick = np.multiply(Etot0_2d_x, nl_kick_array)
        Etot0_2d_y_nl_kick = np.multiply(Etot0_2d_y, nl_kick_array)

        # return to SRW wavefront form
        ex_real = np.real(Etot0_2d_x_nl_kick).flatten(order="C")
        ex_imag = np.imag(Etot0_2d_x_nl_kick).flatten(order="C")

        ey_real = np.real(Etot0_2d_y_nl_kick).flatten(order="C")
        ey_imag = np.imag(Etot0_2d_y_nl_kick).flatten(order="C")

        x = np.linspace(lp_wfr.mesh.xStart, lp_wfr.mesh.xFin, lp_wfr.mesh.nx)
        y = np.linspace(lp_wfr.mesh.yStart, lp_wfr.mesh.yFin, lp_wfr.mesh.ny)

        # remake the wavefront
        thisSlice.wfr = srwutil.make_wavefront(
            ex_real,
            ex_imag,
            ey_real,
            ey_imag,
            thisSlice.photon_e_ev,
            x,
            y,
        )

        return thisSlice


def _interp_to_odd(x_old, y_old, mesh_old):

    nx, ny = len(x_old), len(y_old)
    if nx % 2 == 0:
        x_new = np.linspace(np.min(x_old), np.max(x_old), nx + 1)
    else:
        x_new = np.copy(x_old)
    if ny % 2 == 0:
        y_new = np.linspace(np.min(y_old), np.max(y_old), ny + 1)
    else:
        y_new = np.copy(y_old)

    if nx % 2 == 0 or ny % 2 == 0:
        mesh_new = {}
        for mesh in mesh_old:
            pre_interp = mesh_old["{}".format(mesh)]
            rect_biv_spline = RectBivariateSpline(x_old, y_old, pre_interp)
            post_interp = rect_biv_spline(x_new, y_new)
            mesh_new["{}".format(mesh)] = post_interp
    else:
        mesh_new = copy.deepcopy(mesh_old)

    return x_new, y_new, mesh_new
