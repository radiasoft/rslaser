### thermooptic.py
### June 2023

from fenics import *
from h5py import File
from numpy import array, zeros, pi
from scipy.special import gamma, expi
from mshr import Cylinder, generate_mesh

class ThermoOptic:
    u"""
    Describes thermo-optic effects in Crystal optical elements due to laser heating
    
    Args:
    * `crystal`- Crystal object to perform simulations & calculations for
    * `heat_load`- profile of the heat load deposited by the pumping laser
    * `bc_tol`- tolerance for determining a point presence at boundary
    * `mesh_density`- density of mesh points used by FEniCS solver
    """
    
    # C-style expressions defining heat load profiles & their input arguments for use by FEniCS
    #   - Gaussian expression is from Innocenzi et al (1990), doi:10.1063/1.103083
    #   - Higher-order Gaussian ("HOG") expression is from Schmid, Graf, & Weber (2000), doi:10.1364/JOSAB.17.001398
    #   - Tophat expression is from Chen et al (1997), doi:10.1109/3.605566
    
    PROFILES = {
        "gaussian" : "dT*exp(-2*(x[0]*x[0]+x[1]*x[1])/(wp*wp))*exp(-alpha*(x[2]-z0))",
        "hog" : "dT*exp(-pow((x[0]*x[0]+x[1]*x[1])/(2*wp*wp),Order))*exp(-alpha*(x[2]-z0))",
        "tophat" : "(x[0]*x[0]+x[1]*x[1]<=wp)? dT*alpha*exp(-alpha*(x[2]-z0))/(1-exp(2*alpha*z0)) : 0",
    }

    # Expressions for wavelength- & temperature-dependent indices of refraction
    #   - n0 values (first terms) are from RP Photonics, www.rp-photonics.com/
    #   - nT expression for Al203 are from Tapping & Reilly (1986), doi:10.1364/JOSAA.3.000610
    #   - nT expression for NdYAG are from Brown (1998), doi:10.1109/3.736113
    
    INDICES = {
        "Al2O3" : lambda T: 1.75991 + 1.28e-5*T + 3.1e-9*T**2,
        "NdYAG": lambda T: 1.82 + (-2.59e-6+2.61e-8*T+6.02e-11*T**2)*T,
    }
    
    # Boundary condition types
    BCTYPES = {"dirichlet": DirichletBC, }
    
    __slots__ = ("mesh", "space", "crystal", "heat_load", "boundary", "eval_points")
    
    def __init__(self, crystal, mesh_density=25):
        
        # Attempt to generate Crystal mesh, & solution space
        try:
            self.crystal = crystal
            L = self.crystal.length
            r = self.crystal.radius
            self.mesh = generate_mesh(Cylinder(Point(0.,0.,L/2.), Point(0.,0.,-L/2.), r, r), mesh_density)
            self.space = FunctionSpace(self.mesh, 'P', 1)
        except Exception as e:
            raise RuntimeError("Unable to generate Crystal mesh, received error:\n\t-> {:s}".format(str(e)))
            
        # Initialize attributes for heat load, boundary conditions, & evaluation points
        self.heat_load = None
        self.boundary = None
        self.eval_points = array([0., 0., -L/2.])
        
    def set_load(self, heat_load):
        u"""
        Sets the heat load profile
        
        Args:
        * `heat_load`- str designating a heat load profile or a FEniCS Expression/UserExpression
        """
        
        # Validate choice of heat load & directly set if a FEniCS Expression/UserExpression was given
        if not isinstance(heat_load, (str, Expression, UserExpression)):
            raise ValueError("\'heat_load\' must be a str or FEniCS Expression/UserExpression")
        elif isinstance(heat_load, (Expression, UserExpression)):
            self.heat_load = heat_load
            return
        elif profile.lower() not in self.PROFILES:
            raise ValueError("\'heat_load\' (str) must be one of: {:s}".format(", ".join(self.PROFILES)))
        
        # Define shortnames for useful quantities
        Kc = self.crystal.Kc
        L = self.crystal.length
        alpha = self.crystal.alpha
        wp = self.crystal.pump_waist
        Pabs = self.crystal.pump_energy*self.crystal.pump_rate

        # Define & compute heat load parameters
        heat_params = {"wp":wp, "alpha":alpha, "z0":-L/2.}
        Pth = Pabs*abs(1.-crystal.lambda_pump/crystal.lambda_seed)
        if profile in ("uniform", "tophat"):
            Vol = pi*wp**2*(1.-exp(-alpha*L))/alpha
        elif profile=="gaussian":
            Vol = 0.5*pi*wp**2*(1.-exp(-alpha*L))/alpha
        elif profile=="hog":
            order = self.crystal.pump_order
            heat_params["Order"] = order
            Gint = 2.**(1.-2./order)*wp**2.*gamma(2./order)/order
            Vol = pi*Gint*(1.-exp(-alpha*L))/alpha
        heat_params["dT"] = Pth/(Vol*Kc)
        
        self.heat_load = Expression(self.PROFILES[heat_load], degree=1, **heat_params)
        
    def set_boundary(self, bc_tol, bc_type="dirichlet"):
        u"""
        Sets the boundary conditions for thermo-optic calculations
        
        Args:
        * `bc_tol`- boundary tolerance (m)
        * `bc_type`- boundary condition type (default Dirichlet)
        """
        
        # Validate choice of boundary condition
        if bc_type not in self.BCTYPES:
            raise NotImplementedError("\'bc_type\' must be one of: {:s}".format(", ".join(self.BCTYPES)))
        BC = self.BCTYPES[bc_type]
        if (not isinstance(bc_tol, float)) or (bc_tol<=0):
            raise ValueError("\'bc_tol\' must be a float greater than zero")
        boundary = lambda x, on_boundary: on_boundary and near(x[0]*x[0]+x[1]*x[1], r**2., bc_tol)
        self.boundary = BC(self.space, Constant(self.crystal.Tc), boundary)
        
    def set_points(self, npts, edge=.98):
        u"""
        Sets temperature evaluation points in cylindrical coordinates
        
        Args:
        * npts- number of grid points along each cylindrical axis (r, w, and z)
        * `edge`- fraction of radial extent to use during evaluation (default 0.98)
        """
        
        # Validate choice of evaluation point numbers & edge
        if (not isinstance(npts, (tuple, list))) | (len(npts)!=3):
            raise ValueError("\'npts\' must be a list or tuple of 3 integers")
        if edge<=0 | edge>1:
            raise ValueError("\'edge\' must be a number in the range (0, 1]")
        nr, nw, nz = npts
        if not (nr or nz):
            raise ValueError("number of points along either radial or longitudinal axis must be non-zero")

        # Construct cylindrical grid of evaluation points
        rs = edge*r*array(range(1,nx+1))/nx
        zs = edge*(-L/2.+L*array(range(nz)))/nz
        if nw:
            ws = 2.*pi*array(range(nw))/nw
            self.eval_points = array([[[0.,0.,z]]+[[r*cos(om),r*sin(om),z] for r in rs for om in ws] for z in zs]).reshape((nz*(nw*nr+1),3))
        else:
            self.eval_points = array([[[0.,0.,z]]+[[r,0.,z] for r in rs] for z in zs]).reshape((nz*(nr+1),3))
                            
    def solve_time(self, runtime, dt=1e-3, load_off=None, save=False, path="./T-crystal.h5"):
        u"""
        Solves the fully time-dependent heat equation for the Crystal
        
        Given in Eqn. 2.1.1 in Chenais et al (2006), doi:10.1016/j.pquantelec.2006.12.001
        """
        
        # Ensure that a heat load, boundary conditions, & evaluation points have been set
        if (not self.heat_load) | (not self.boundary) | (not self.eval_points):
            raise RuntimeError("must set heat load, boundary conditions, & evaluation points prior to simulation")
        
        # Set FEniCS log & define shortnames for useful quantities
        set_log_level(50)
        f = self.heat_load
        ad = self.crytal.Kc/(self.crystal.cp*self.crystal.rho)

        # Define time parameters & create output temperature array
        Nt = int(runtime/dt)+1
        npts = self.eval_points.shape[0]
        Ts = zeros((Nt, npts))
        
        # Set initial temperature state
        T = interpolate(self.crystal.Tc, self.space)
        Ts[0,:] = [T(pt) for pt in self.eval_points]
            
        # Initialize variational variables used by FEniCS
        u = TrialFunction(self.space)
        v = TestFunction(self.space)
        T_solve = Function(self.space)
        
        # Determine number of steps with/without thermal loading
        if load_off!=None: load_off = int(load_off/dt)
        else: load_off = Nt
        
        # Define time-dependent differential equation for temperature
        F = u*v*dx+dt*ad*dot(grad(u),grad(v))*dx-(T+dt*f)*v*dx
        Fl, Fr = lhs(F), rhs(F)

        # Solve the differential equation under thermal loading
        for n in range(1,load_off):
            solve(Fl==Fr, T_solve, self.boundary)
            T.assign(T_solve)
            Ts[n,:] = [T(pt) for pt in self.eval_points]
        
        # Solve the differential equation without thermal loading for remaining steps (if any) 
        if load_off<Nt:
            F = u*v*dx+dt*ad*dot(grad(u),grad(v))*dx
            Fl, Fr = lhs(F), rhs(F)
            for n in range(load_off, Nt):
                solve(Fl==Fr, T_solve, self.boundary)
                T.assign(T_solve)
                Ts[n,:] = [T(pt) for pt in self.eval_points]
        
        # Return temperature field, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts
        
    def solve_steady(self, save=False, path="./T-crystal.h5"):
        u"""
        Solves the steady state (time-independent) heat equation for the Crystal.
        
        Given in Eqn. 2.1.1 (without time term) in Chenais et al (2006), doi:10.1016/j.pquantelec.2006.12.001
        """
        
        # Ensure that a heat load, boundary conditions, & evaluation points have been set
        if (not self.heat_load) | (not self.boundary) | (not self.eval_points):
            raise RuntimeError("must set heat load, boundary conditions, & evaluation points prior to simulation")
        
        # Set FEniCS log & define shortnames for useful quantities
        set_log_level(50)
        f = self.heat_load
        T = Function(self.space)
            
        # Initialize variational variables used by FEniCS
        u = TrialFunction(self.space)
        v = TestFunction(self.space)
                
        # Define time-independent differential equation for temperature
        F = dot(grad(u),grad(v))*dx - f*v*dx
        solve(F==0.0, T, self.boundary)
        Ts = [T(pt) for pt in self.eval_points]
        
        # Return temperature field, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts
    
    def gaussian_solution(self, save=False, path="./T-crystal.h5"):
        u"""
        Computes a solution to the steady state heat equation given a Gaussian heat load
        
        Given in Eqn. 7 of Innocenzi et al (1990), doi:10.1063/1.103083
        """
        
        # Define shortnames for useful quantities
        Kc = self.crystal.Kc
        L = self.crystal.length
        r0 = self.crystal.radius
        alpha = self.crystal.alpha
        wp = self.crystal.pump_waist
        Pabs = self.crystal.pump_energy*self.crystal.pump_rate
        Pth = Pabs*abs(1.-crystal.lambda_pump/crystal.lambda_seed)

        # Evaluate the steady state solution for Gaussian heat load at all evaluation points
        zs = self.eval_pts[:,2]
        rs = (self.eval_pts[:,0]**2+self.eval_pts[:,1]**2)**.5
        Ts = Pth/(4*pi*Kc)*alpha*exp(-alpha*(zs+L/2.))*(2*log(r0/rs)+expi(2*(r0/wp)**2)+expi(2*(rs/wp)**2))
        
        # Return temperature field, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts
        
    def hog_solution(self, save=False, path="./T-crystal.h5"):
        u"""
        Computes a solution to the steady state heat equation given a higher-order Gaussian heat load
        
        Given in Eqn. ? of Schmid (2000), doi:10.1364/JOSAB.17.001398
        """
        
        # Define shortnames for useful quantities
        Kc = self.crystal.Kc
        L = self.crystal.length
        r0 = self.crystal.radius
        alpha = self.crystal.alpha
        wp = self.crystal.pump_waist
        Pabs = self.crystal.pump_energy*self.crystal.pump_rate
        Pth = Pabs*abs(1.-crystal.lambda_pump/crystal.lambda_seed)
        
        # Compute steady state temperature change at crystal face
        Gint = 2.**(1.-2./order)*wp**2.*gamma(2./order)/order
        Vol = pi*Gint*(1.-exp(-alpha*L))/alpha
        dT = Pth/(Vol * Kc)
        
        # Evaluate the steady state solution for uniform heat load at all evaluation points
        zs = self.eval_pts[:,2]
        rs = (self.eval_pts[:,0]**2+self.eval_pts[:,1]**2)**.5
        Ts = dT*exp(-2.*(rs/wp)**order)*exp(-alpha*(zs+L/2.))
        
        # Return temperature field, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts
    
    def tophat_solution(self, save=False, path="./T-crystal.h5"):
        u"""
        Computes a solution to the steady state heat equation given a tophat heat load 
        
        Given in Eqn. 13 of Chen et al (1997), doi:10.1109/3.605566
        (Also Eqn. 2.1.6 of Chenais et al (2006), doi:10.1016/j.pquantelec.2006.12.001)
        """
        
        # Define shortnames for useful quantities
        Kc = self.crystal.Kc
        L = self.crystal.length
        r0 = self.crystal.radius
        alpha = self.crystal.alpha
        order = self.crystal.pump_order
        Pabs = self.crystal.pump_energy*self.crystal.pump_rate
        Pth = Pabs*abs(1.-crystal.lambda_pump/crystal.lambda_seed)

        # Evaluate the steady state solution for uniform heat load at all evaluation points
        zs = self.eval_pts[:,2]
        rs = (self.eval_pts[:,0]**2+self.eval_pts[:,1]**2)**.5
        Ts = Pth/(4*pi*Kc)*alpha*exp(-alpha*(zs+L/2.))/(1-exp(-alpha*L))
        Ts[rs<=wp] *= 2*log(r0/wp)+1-(r/wp)**2
        Ts[rs>wp] *= 2*log(r0/r)
        
        # Return temperature field, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset(data=Ts)
        return Ts
                             
    def compute_indices(self, Ts, material="Ti:Al2O3", fit_width=None, save=False, path="./n-crystal.h5"):
        u"""
        Computes indices of refraction based on material & temperature
        
        Args:
        * `Ts`- temperatures to use for calculations
        * `material`- material type (Al203 or NdYAG, default Ti:Al2O3)
        * `fit_edge`- width of central region used in curve fitting (m)
        """
        
        # Use default fit_width relative to pump waist if none given
        if not fit_width:
            fit_width = .5*self.crystal.pump_waist
        
        # Define temperature dependent index & fitting functions
        nfun = self.INDICES[material]
        quad_fit = lambda x, A, B: -0.5*A * x**2.0 + B
        
        # Define short names for useful quantities
        zs = self.eval_pts[:,2]
        rs = (self.eval_pts[:,0]**2+self.eval_pts[:,1]**2)**.5
        in_fit = abs(rs)<=fit_width
        Trwz = Ts.reshape(Ts.shape, order='F')
        
        # Compute analytical & quadratic fit index values
        nTrz = zeros((len(zs), len(rs)))
        nFit = zeros((len(zs), 2, 2))
        for z in range(nz):
            nTrz[z] = array([nfun(T) for T in Trz[:,z]])
            pfit, varfit = curve_fit(quad_fit, rs[in_fit], nTrz[z,in_fit], 2, cov=True)
            nFit[z,0,:] = pfit
            nFit[z,1,:] = diag(varfit)
            
        # Return index values according to pumping direction
        if self.crystal.pump_type == "right":
            nTrz = nTrz[::-1]
            nFit = nFit[::-1]
        elif self.params.pop_inversion_pump_type == "dual":
            nTrz = (nTrz+nTrz[::-1])/2.
            nFit = (nFit+nFit[::-1])/2.
        
        # Return indices, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset("nTrz", data=Ts)
                h5File.create_dataset("nFit", data=Ts)
        return nTrz, nFit
        
        '''
        # fix negative n2 vals and ****divide through by 2 based on Gaussian duct definition n(r) = n0 - 1/2*n2*r^2**** - see rp-photonics.com
        n2_full_array = np.multiply(n2_full_array, -2.0)
        n2_full_array = np.multiply(n2_full_array, 1.0e4)

        # Look at why n2 goes negative...
        n2_full_array[n2_full_array <= 0.0] = 1.0e-6
        
        z_full_array = np.linspace(0.0, self.length, len(n0_full_array))
        n0_fit = splrep(z_full_array, n0_full_array)
        n2_fit = splrep(z_full_array, n2_full_array)

        z_crystal_slice = (self.length / self.nslice) * (np.arange(self.nslice) + 0.5)
        n0_slice_array = splev(z_crystal_slice, n0_fit)
        n2_slice_array = splev(z_crystal_slice, n2_fit)

        if set_n:
            for s in self.slice:
                s.n0 = n0_output[s.slice_index]
                s.n2 = n2_output[s.slice_index]
        '''
                             
    def compute_ABCD(self, ns, save=False, path="./ABCD-crystal.h5"):
        u"""
        Computes ABCD matrices for each longitudinal eval point in the crystal
        
        Args:
        * `ns`- array of fitted n0/n2 values
        """
        
        # Define shortnames for useful quantities
        nz = len(self.eval_pts[:,2])
        dz = self.crystal.length/nz
        
        # Compute ABCD matrices at each longitudinal point
        ABCDs = zeros((nz, 2, 2))+0j
        for z in range(nz):
            n2, n0 = ns[z,0]
            gamma = (n2/n0+0j)**.5
            ABCDs[z] = [[cos(gamma*dz), sin(gamma*dz)/(n0*gamma)],[-n0*gamma*sin(gamma*dz), cos(gamma*dz)]]
            
        # Compute total ABCD matrix
        full_ABCD = ABCD[::-1].prod(dim=0)
        
        # Return ABCD matrices, saving if requested
        if save: 
            with File(path, "w") as h5File:
                h5File.create_dataset("ABCDs", data=ABCDs)
                h5File.create_dataset("full_ABCD", data=full_ABCD)
        return ABCDs, full_ABCD
    