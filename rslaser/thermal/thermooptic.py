### thermooptic.py
### June 2023

from fenics import *
from h5py import File
from mpmath import hyp2f2
from scipy.optimize import curve_fit
from mshr import Cylinder, generate_mesh
from scipy.special import gamma, gammaincc as GammaI, exp1
from numpy import array, zeros, pi, exp, log, unique, diag, argsort, sin, cos, sinc

set_log_level(30)


class ThermoOptic:
    """
    Describes thermo-optic effects in Crystal optical elements due to laser heating

    Args:
    * `crystal`- Crystal object to perform simulations & calculations for
    * `heat_load`- profile of the heat load deposited by the pumping laser
    * `bc_tol`- tolerance for determining a point presence at boundary
    * `mesh_density`- density of mesh points used by FEniCS solver
    """

    # C-style expressions defining heat load profiles & their input arguments for use by FEniCS
    #   - Gaussian expression is from Innocenzi et al (1990), doi:10.1063/1.103083
    #   - Higher-order Gaussian ("HOG") expression is from Cini & Mackenzie (2017), doi:10.1007/s00340-017-6848-y
    #       ~ This paper also contains good information about Gaussian & Tophat profiles
    #   - Tophat expression is from Chen et al (1997), doi:10.1109/3.605566

    PROFILES = {
        "tophat": "(x[0]*x[0]+x[1]*x[1]<=wp*wp)? Q0*exp(-alpha*(x[2]-z0)) : 0",
        "gaussian": "Q0*exp(-2*(x[0]*x[0]+x[1]*x[1])/(wp*wp))*exp(-alpha*(x[2]-z0))",
        "hog": "Q0*exp(-2*pow((x[0]*x[0]+x[1]*x[1])/(wp*wp),P/2.))*exp(-alpha*(x[2]-z0))",
    }

    # Expressions for wavelength- & temperature-dependent indices of refraction
    #   - n0 values (first terms) are from RP Photonics, www.rp-photonics.com/
    #   - nT expression for Al203 are from Tapping & Reilly (1986), doi:10.1364/JOSAA.3.000610
    #   - nT expression for NdYAG are from Brown (1998), doi:10.1109/3.736113

    INDICES = {
        "Ti:Al2O3": lambda T: 1.75991 + 1.28e-5 * T + 3.1e-9 * T**2,
        "NdYAG": lambda T: 1.82 + (-2.59e-6 + 2.61e-8 * T + 6.02e-11 * T**2) * T,
    }

    # Boundary condition types
    BCTYPES = {
        "dirichlet": DirichletBC,
    }

    __slots__ = ("mesh", "space", "crystal", "heat_load", "boundary", "eval_pts")

    def __init__(self, crystal, mesh_density=50):

        # Attempt to generate Crystal mesh, & solution space
        try:
            self.crystal = crystal
            L = self.crystal.length * 1.0e2
            r0 = self.crystal.radius * 1.0e2
            self.mesh = generate_mesh(
                Cylinder(Point(0.0, 0.0, L / 2.0), Point(0.0, 0.0, -L / 2.0), r0, r0),
                mesh_density,
            )
            self.space = FunctionSpace(self.mesh, "P", 1)
        except Exception as e:
            raise RuntimeError(
                "Unable to generate Crystal mesh, received error:\n\t-> {:s}".format(
                    str(e)
                )
            )

        # Initialize attributes for heat load, boundary conditions, & evaluation points
        self.heat_load = None
        self.boundary = None
        self.eval_pts = array([])

    def set_load(self, heat_load):
        """
        Sets the heat load profile

        Volume integrals given in Eqns. 10, 17, & 26 of Cini & Mackenzie (2017), doi:10.1007/s00340-017-6848-y

        Args:
        * `heat_load`- str designating a heat load profile or a FEniCS Expression/UserExpression
        """

        # Validate choice of heat load & directly set if a FEniCS Expression/UserExpression was given
        if not isinstance(heat_load, (str, Expression, UserExpression)):
            raise ValueError(
                "'heat_load' must be a str or FEniCS Expression/UserExpression"
            )
        elif isinstance(heat_load, (Expression, UserExpression)):
            self.heat_load = heat_load
            return
        elif heat_load.lower() not in self.PROFILES:
            raise ValueError(
                "'heat_load' (str) must be one of: {:s}".format(
                    ", ".join(self.PROFILES)
                )
            )

        # Define shortnames for useful quantities
        Kc = self.crystal.Kc / 1.0e2
        L = self.crystal.length * 1.0e2
        r0 = self.crystal.radius * 1.0e2
        alpha = self.crystal.alpha / 1.0e2
        wp = self.crystal.params.pop_inversion_pump_waist * 1.0e2
        Pabs = (
            self.crystal.params.pop_inversion_pump_energy
            * self.crystal.params.pop_inversion_pump_rep_rate
        )
        lambda_seed = self.crystal.params.pop_inversion_lambda_seed
        lambda_pump = self.crystal.params.pop_inversion_lambda_pump

        # Define parameters related to crystal & pump parameters
        heat_params = {"wp": wp, "alpha": alpha, "z0": -L / 2.0}

        # Compute fractional thermal load & absorption efficiency
        etah = abs(1.0 - lambda_pump / lambda_seed)
        etaAbs = 1.0 - exp(-alpha * L)

        # Compute pumping volume & heat load constant
        if heat_load == "tophat":
            Vol = pi * wp**2 * etaAbs / alpha

        elif heat_load == "gaussian":
            Vol = 0.5 * pi * wp**2 * (1 - exp(-2 * (r0 / wp) ** 2)) * etaAbs / alpha

        elif heat_load == "hog":
            order = self.crystal.params.pop_inversion_pump_gaussian_order
            heat_params["P"] = order
            Gams = GammaI(2.0 / order, 0.0) - GammaI(
                2.0 / order, 2.0 * (r0 / wp) ** order
            )
            Vol = (
                (2.0 * pi * wp**2 / order)
                * 4.0 ** (-1.0 / order)
                * Gams
                * etaAbs
                / alpha
            )

        heat_params["Q0"] = etah * Pabs / (Kc * Vol)

        # Handling different cases for the pump type, set heat load expression
        if self.crystal.params.pop_inversion_pump_type == "right":
            hl_expr = self.PROFILES[heat_load].replace("x[2]-z0", "-x[2]-z0")
        elif self.crystal.params.pop_inversion_pump_type == "dual":
            hl_expr = self.PROFILES[heat_load].replace("x[2]-z0", "-abs(x[2])-z0")
        else:
            hl_expr = self.PROFILES[heat_load]
        self.heat_load = Expression(hl_expr, degree=1, **heat_params)

    def set_boundary(self, bc_tol=0.1, bc_type="dirichlet"):
        """
        Sets the boundary conditions for thermo-optic calculations

        Args:
        * `bc_tol`- boundary tolerance (cm, default 0.1)
        * `bc_type`- boundary condition type (default Dirichlet)
        """

        r0 = self.crystal.radius * 1.0e2

        # Validate choice of boundary condition
        if bc_type not in self.BCTYPES:
            raise NotImplementedError(
                "'bc_type' must be one of: {:s}".format(", ".join(self.BCTYPES))
            )
        BC = self.BCTYPES[bc_type]
        if (not isinstance(bc_tol, float)) or (bc_tol <= 0):
            raise ValueError("'bc_tol' must be a float greater than zero")
        boundary = lambda x, on_boundary: on_boundary and near(
            x[0] * x[0] + x[1] * x[1], r0 * r0, bc_tol
        )
        self.boundary = BC(self.space, Constant(self.crystal.params.Tc), boundary)

    def set_points(self, npts, edge=0.98):
        """
        Sets temperature evaluation points in cylindrical coordinates

        Args:
        * npts- number of grid points along each cylindrical axis (r, w, and z)
        * `edge`- fraction of radial extent to use during evaluation (default 0.98)
        """

        # Validate choice of evaluation point numbers & edge
        if (not isinstance(npts, (tuple, list))) | (len(npts) != 3):
            raise ValueError("'npts' must be a list or tuple of 3 integers")
        if (edge <= 0) | (edge > 1):
            raise ValueError("'edge' must be a number in the range (0, 1]")
        nr, nw, nz = npts
        if not (nr or nz):
            raise ValueError(
                "number of points along either radial or longitudinal axis must be non-zero"
            )

        # Define shortnames for useful quantities
        L = self.crystal.length * 1.0e2
        r0 = self.crystal.radius * 1.0e2

        # Construct cylindrical grid of evaluation points
        rs = edge * r0 * array(range(1, nr + 1)) / (nr - 1)
        zs = edge * (-L / 2.0 + L * array(range(nz)) / (nz - 1))
        if nw:
            ws = 2.0 * pi * array(range(nw)) / nw
            self.eval_pts = array(
                [
                    [[0.0, 0.0, z]]
                    + [[r * cos(om), r * sin(om), z] for r in rs for om in ws]
                    for z in zs
                ]
            ).reshape((nz * (nw * nr + 1), 3))
        else:
            self.eval_pts = array(
                [[[0.0, 0.0, z]] + [[r, 0.0, z] for r in rs] for z in zs]
            ).reshape((nz * (nr + 1), 3))

    def solve_time(
        self, runtime, dt=1e-3, load_off=None, save=False, path="./T-crystal.h5"
    ):
        """
        Solves the fully time-dependent heat equation for the Crystal

        Given in Eqn. 2.1.1 in Chenais et al (2006), doi:10.1016/j.pquantelec.2006.12.001
        """

        # Ensure that a heat load, boundary conditions, & evaluation points have been set
        if (not self.heat_load) | (not self.boundary) | (not self.eval_pts.size):
            raise RuntimeError(
                "must set heat load, boundary conditions, & evaluation points prior to simulation"
            )

        # Set FEniCS log & define shortnames for useful quantities
        set_log_level(50)
        f = self.heat_load
        ad = self.crytal.Kc / (self.crystal.cp * self.crystal.rho)

        # Define time parameters & create output temperature array
        Nt = int(runtime / dt) + 1
        npts = self.eval_pts.shape[0]
        Ts = zeros((Nt, npts))

        # Set initial temperature state
        T = interpolate(self.crystal.params.Tc, self.space)
        Ts[0, :] = array([T(pt) for pt in self.eval_pts])

        # Initialize variational variables used by FEniCS
        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        T_solve = Function(self.space)

        # Determine number of steps with/without thermal loading
        if load_off != None:
            load_off = int(load_off / dt)
        else:
            load_off = Nt

        # Define time-dependent differential equation for temperature
        F = u * v * dx + dt * ad * dot(grad(u), grad(v)) * dx - (T + dt * f) * v * dx
        Fl, Fr = lhs(F), rhs(F)

        # Solve the differential equation under thermal loading
        for n in range(1, load_off):
            solve(Fl == Fr, T_solve, self.boundary)
            T.assign(T_solve)
            Ts[n, :] = [T(pt) for pt in self.eval_pts]

        # Solve the differential equation without thermal loading for remaining steps (if any)
        if load_off < Nt:
            F = u * v * dx + dt * ad * dot(grad(u), grad(v)) * dx
            Fl, Fr = lhs(F), rhs(F)
            for n in range(load_off, Nt):
                solve(Fl == Fr, T_solve, self.boundary)
                T.assign(T_solve)
                Ts[n, :] = [T(pt) for pt in self.eval_pts]

        # Return temperature field, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts

    def solve_steady(self, save=False, path="./T-crystal.h5"):
        """
        Solves the steady state (time-independent) heat equation for the Crystal.

        Given in Eqn. 2.1.1 (without time term) in Chenais et al (2006), doi:10.1016/j.pquantelec.2006.12.001
        """

        # Ensure that a heat load, boundary conditions, & evaluation points have been set
        if (not self.heat_load) | (not self.boundary) | (not self.eval_pts.size):
            raise RuntimeError(
                "must set heat load, boundary conditions, & evaluation points prior to simulation"
            )

        # Set FEniCS log & define shortnames for useful quantities
        set_log_level(50)
        f = self.heat_load

        # Initialize variational variables used by FEniCS
        u = Function(self.space)
        v = TestFunction(self.space)

        # Define time-independent differential equation for temperature
        F = dot(grad(u), grad(v)) * dx - f * v * dx
        solve(F == 0.0, u, self.boundary)
        Ts = array([u(pt) for pt in self.eval_pts])

        # Return temperature field, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts

    def tophat_solution(self, save=False, path="./T-crystal.h5"):
        """
        Computes a solution to the steady state heat equation given a tophat heat load

        Given in Eqn. 13 of Chen et al (1997), doi:10.1109/3.605566
        (Also Eqn. 2.1.6 of Chenais et al (2006), doi:10.1016/j.pquantelec.2006.12.001)
        """

        # Define shortnames for useful quantities
        Kc = self.crystal.Kc / 1.0e2
        L = self.crystal.length * 1.0e2
        r0 = self.crystal.radius * 1.0e2
        alpha = self.crystal.alpha / 1.0e2
        wp = self.crystal.params.pop_inversion_pump_waist * 1.0e2
        Pabs = (
            self.crystal.params.pop_inversion_pump_energy
            * self.crystal.params.pop_inversion_pump_rep_rate
        )
        lambda_seed = self.crystal.params.pop_inversion_lambda_seed
        lambda_pump = self.crystal.params.pop_inversion_lambda_pump
        z0 = -L / 2.0

        # Compute fractional thermal load & absorption efficiency
        etah = abs(1.0 - lambda_pump / lambda_seed)
        etaAbs = 1.0 - exp(-alpha * L)

        # Extract radial & axial values for all evaluation points
        zs = self.eval_pts[:, 2]
        rs = (self.eval_pts[:, 0] ** 2 + self.eval_pts[:, 1] ** 2) ** 0.5
        if self.crystal.params.pop_inversion_pump_type == "right":
            zs *= -1
        elif self.crystal.params.pop_inversion_pump_type == "dual":
            zs = -abs(zs)

        # Evaluate the analytical steady state solution for a tophat heat load
        Trz = etah * (Pabs / etaAbs) * alpha / (4 * pi * Kc) * exp(-alpha * (zs - z0))
        Trz[rs <= wp] *= 2 * log(r0 / wp) + 1 - (rs[rs <= wp] / wp) ** 2
        Trz[rs > wp] *= 2 * log(r0 / rs[rs > wp])

        # Return temperature field, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Trz)
        return Trz

    def gaussian_solution(self, save=False, path="./T-crystal.h5"):
        """
        Computes a solution to the steady state heat equation given a Gaussian heat load

        Given in Eqn. 7 of Innocenzi et al (1990), doi:10.1063/1.103083
        """

        # Define shortnames for useful quantities
        Kc = self.crystal.Kc / 1.0e2
        L = self.crystal.length * 1.0e2
        r0 = self.crystal.radius * 1.0e2
        alpha = self.crystal.alpha / 1.0e2
        wp = self.crystal.params.pop_inversion_pump_waist * 1.0e2
        Pabs = (
            self.crystal.params.pop_inversion_pump_energy
            * self.crystal.params.pop_inversion_pump_rep_rate
        )
        lambda_seed = self.crystal.params.pop_inversion_lambda_seed
        lambda_pump = self.crystal.params.pop_inversion_lambda_pump
        z0 = -L / 2.0

        # Compute fractional thermal load & absorption efficiency
        etah = abs(1.0 - lambda_pump / lambda_seed)
        etaAbs = 1.0 - exp(-alpha * L)

        # Extract radial & axial values for all evaluation points
        zs = self.eval_pts[:, 2]
        rs = (self.eval_pts[:, 0] ** 2 + self.eval_pts[:, 1] ** 2) ** 0.5
        rs[rs == 0] = 1e-6  # Prevents a divide by zero error in the log term below
        if self.crystal.params.pop_inversion_pump_type == "right":
            zs *= -1
        elif self.crystal.params.pop_inversion_pump_type == "dual":
            zs = -abs(zs)

        # Evaluate the analytical steady state solution for a Gaussian heat load
        Trz = etah * (Pabs / etaAbs) * alpha / (4 * pi * Kc) * exp(-alpha * (zs - z0))
        Trz *= log((r0 / rs) ** 2) + exp1(2 * (r0 / wp) ** 2) - exp1(2 * (rs / wp) ** 2)

        # Return temperature field, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Trz)
        return Trz

    def hog_solution(self, save=False, path="./T-crystal.h5"):
        """
        Computes a solution to the steady state heat equation given a higher-order Gaussian heat load

        Given in Eqn. 28 of Cini & Mackenzie (2017), doi:10.1007/s00340-017-6848-y
        """

        # Define shortnames for useful quantities
        Kc = self.crystal.Kc / 1.0e2
        L = self.crystal.length * 1.0e2
        r0 = self.crystal.radius * 1.0e2
        alpha = self.crystal.alpha / 1.0e2
        order = self.crystal.params.pop_inversion_pump_gaussian_order
        wp = self.crystal.params.pop_inversion_pump_waist * 1.0e2
        Pabs = (
            self.crystal.params.pop_inversion_pump_energy
            * self.crystal.params.pop_inversion_pump_rep_rate
        )
        lambda_seed = self.crystal.params.pop_inversion_lambda_seed
        lambda_pump = self.crystal.params.pop_inversion_lambda_pump
        z0 = -L / 2.0

        # Compute fractional thermal load & absorption efficiency
        etah = abs(1.0 - lambda_pump / lambda_seed)
        etaAbs = 1.0 - exp(-alpha * L)

        # Extract radial & axial values for all evaluation points
        zs = self.eval_pts[:, 2]
        rs = (self.eval_pts[:, 0] ** 2 + self.eval_pts[:, 1] ** 2) ** 0.5
        if self.crystal.params.pop_inversion_pump_type == "right":
            zs *= -1
        elif self.crystal.params.pop_inversion_pump_type == "dual":
            zs = -abs(zs)

        # Evaluate the analytical steady state solution for a Gaussian heat load
        a = 2.0 / order
        hyp1 = r0**2 * float(hyp2f2(a, a, a + 1, a + 1, -2 * (r0 / wp) ** order))
        hyp2 = array(
            [
                r**2 * float(hyp2f2(a, a, a + 1, a + 1, -2 * (r / wp) ** order))
                for r in rs
            ]
        )
        Trz = (
            etah
            * (Pabs / etaAbs)
            * alpha
            / (4 * pi * Kc)
            * exp(-alpha * (zs - z0))
            * (hyp1 - hyp2)
        )
        Trz *= 2.0 ** (a - 1.0) * order / (gamma(a) * wp**2)

        # Return temperature field, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Trz)
        return Trz

    def compute_indices(
        self, Ts, material="Ti:Al2O3", fit_width=None, save=False, path="./n-crystal.h5"
    ):
        """
        Computes indices of refraction based on material & temperature

        Args:
        * `Ts`- temperatures to use for calculations
        * `material`- material type (Al203 or NdYAG, default Ti:Al2O3)
        * `fit_width`- width of central region used in curve fitting (m)
        """

        # Use default fit_width relative to pump waist if none given
        if not fit_width:
            fit_width = 0.5 * self.crystal.params.pop_inversion_pump_waist

        # Define short names for useful quantities
        zs = unique(self.eval_pts[:, 2]) / 1.0e2
        rs = (self.eval_pts[:, 0] ** 2 + self.eval_pts[:, 1] ** 2) ** 0.5 / 1.0e2
        npts = len(self.eval_pts)

        # Compute analytical & quadratic fit index values
        nT = self.INDICES[material](Ts)
        nFit = zeros((len(zs), 2, 2))
        for z in range(len(zs)):
            in_fit = (self.eval_pts[:, 2] / 1.0e2 == zs[z]) * (abs(rs) <= fit_width)
            pfit, varfit = curve_fit(
                lambda r, n0, n2: n0 - 0.5 * n2 * r**2.0, rs[in_fit], nT[in_fit]
            )
            nFit[z, 0, :] = pfit
            nFit[z, 1, :] = diag(varfit)

        # Return indices, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset("nT", data=nT)
                h5File.create_dataset("nFit", data=nFit)
        return nT, nFit

    def compute_ABCD(self, ns, save=False, path="./ABCD-crystal.h5"):
        """
        Computes ABCD matrices for each longitudinal eval point in the crystal

        Args:
        * `ns`- array of fitted n0/n2 values
        """

        # Define shortnames for useful quantities
        nz = len(set(self.eval_pts[:, 2]))
        dz = self.crystal.length / nz

        # Compute ABCD matrices at each longitudinal point
        ABCDs = zeros((nz, 2, 2))
        for z in range(nz):
            n0, n2 = ns[z, 0]
            gamma = (n2 / n0) ** 0.5
            ABCDs[z] = [
                [cos(gamma * dz), dz * sinc(gamma * dz / pi)],
                [-n0 * gamma * sin(gamma * dz), cos(gamma * dz)],
            ]

        # Compute total ABCD matrix
        full_ABCD = ABCDs[-1].copy()
        for ABCD in ABCDs[-2::-1]:
            full_ABCD = full_ABCD @ ABCD

        # Return ABCD matrices, saving if requested
        if save:
            with File(path, "w") as h5File:
                h5File.create_dataset("ABCDs", data=ABCDs)
                h5File.create_dataset("full_ABCD", data=full_ABCD)
        return ABCDs, full_ABCD
