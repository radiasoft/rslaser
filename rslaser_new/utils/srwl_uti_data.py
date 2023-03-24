# -*- coding: utf-8 -*-
"""Data processing functions
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""

import uti_plot_com as srw_io
import numpy as np
import math

# from math import *
from numpy.fft import *
from array import array

import srwlib
from srwlib import srwl
from srwlib import *

import scipy.constants as const


def createGsnSrcSRW(
    sigx,
    sigy,
    num_sig,
    dist_waist,
    pulseE,
    poltype,
    nx=400,
    ny=400,
    phE=10e3,
    mx=0,
    my=0,
):
    """
    This function calculates a Gsn wavefront with waist at zero meters
    and allows calculation at any longitudinal point via dist_waist.
    Args:
        sigx: horizontal beam size at waist [m]
        sigy: vertical beam size at waist [m]
        num_sig: no. of sigmas for Gsn range
        dist_waist: distance of Gsn from waist [m]
        pulseE: energy per pulse [J]
        poltype: polarization type (0=linear horizontal, 1=linear vertical, 2=linear 45 deg, 3=linear 135 deg, 4=circular right, 5=circular left, 6=total)
        nx = no. of horizontal mesh points
        ny = no. of vertical mesh points
        phE: photon energy [eV]
        mx = horizontal Hermite mode
        my = vertical Hermite mode

    Returns:
        wfr
    """

    constConvRad = 1.23984186e-06 / (
        4 * 3.1415926536
    )  ##conversion from energy to 1/wavelength
    rmsAngDiv_x = constConvRad / (phE * sigx)  ##RMS angular divergence [rad]
    rmsAngDiv_y = constConvRad / (phE * sigy)
    sigrL_x = math.sqrt(sigx**2 + (dist_waist * rmsAngDiv_x) ** 2)
    sigrL_y = math.sqrt(sigy**2 + (dist_waist * rmsAngDiv_y) ** 2)

    # Gaussian Beam Source
    GsnBm = SRWLGsnBm()  # Gaussian Beam structure (just parameters)
    GsnBm.x = 0  # Transverse Positions of Gaussian Beam Center at Waist [m]
    GsnBm.y = 0
    GsnBm.z = 0.0  # Longitudinal Position of Waist [m]
    GsnBm.xp = 0  # Average Angles of Gaussian Beam at Waist [rad]
    GsnBm.yp = 0
    GsnBm.avgPhotEn = phE  # Photon Energy [eV]
    GsnBm.pulseEn = pulseE  # Energy per Pulse [J] - to be corrected
    GsnBm.repRate = 1  # Rep. Rate [Hz] - to be corrected
    GsnBm.polar = poltype  # 1- linear horizontal?
    GsnBm.sigX = sigx  # Horiz. RMS size at Waist [m]
    GsnBm.sigY = sigy  # Vert. RMS size at Waist [m]

    GsnBm.sigT = 10e-15  # Pulse duration [s] (not used?)
    GsnBm.mx = mx  # Transverse Gauss-Hermite Mode Orders
    GsnBm.my = my
    # create mesh
    wfr = SRWLWfr()  # Initial Electric Field Wavefront
    wfr.allocate(
        1, nx, ny
    )  # Numbers of points vs Photon Energy (1), Horizontal and Vertical Positions (dummy)
    wfr.mesh.zStart = dist_waist  # Longitudinal Position [m] at which initial Electric Field has to be calculated, i.e. the position of the first optical element
    wfr.mesh.eStart = GsnBm.avgPhotEn  # Initial Photon Energy [eV]
    wfr.mesh.eFin = GsnBm.avgPhotEn  # Final Photon Energy [eV]

    wfr.unitElFld = 1  # Electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)

    distSrc = wfr.mesh.zStart - GsnBm.z
    # Horizontal and Vertical Position Range for the Initial Wavefront calculation
    xAp = num_sig * sigrL_x
    yAp = num_sig * sigrL_y

    wfr.mesh.xStart = -xAp  # Initial Horizontal Position [m]
    wfr.mesh.xFin = xAp  # Final Horizontal Position [m]
    wfr.mesh.yStart = -yAp  # Initial Vertical Position [m]
    wfr.mesh.yFin = yAp  # Final Vertical Position [m]
    # sampFactNxNyForProp = sampFact #sampling factor for adjusting nx, ny (effective if > 0)
    arPrecPar = [
        0
    ]  # sampFact set to zero to allow manual setting of mesh size via wfr.allocate()

    srwl.CalcElecFieldGaussian(wfr, GsnBm, arPrecPar)

    return wfr


def calc_int_from_wfr(_wfr, _pol=6, _int_type=0, _det=None, _fname="", _pr=True):
    # def calc_int_from_wfr(self, _wfr, _pol=6, _int_type=0, _det=None, _fname='', _pr=True):
    """Calculates intensity from electric field and saving it to a file
    :param _wfr: electric field wavefront (instance of SRWLWfr)
    :param _pol: polarization component to extract:
        0- Linear Horizontal;
        1- Linear Vertical;
        2- Linear 45 degrees;
        3- Linear 135 degrees;
        4- Circular Right;
        5- Circular Left;
        6- Total
    :param _int_type: "type" of a characteristic to be extracted:
       -1- No Intensity / Electric Field components extraction is necessary (only Wavefront will be calculated)
        0- "Single-Electron" Intensity;
        1- "Multi-Electron" Intensity;
        2- "Single-Electron" Flux;
        3- "Multi-Electron" Flux;
        4- "Single-Electron" Radiation Phase;
        5- Re(E): Real part of Single-Electron Electric Field;
        6- Im(E): Imaginary part of Single-Electron Electric Field;
        7- "Single-Electron" Intensity, integrated over Time or Photon Energy (i.e. Fluence);
    :param _det: detector (instance of SRWLDet)
    :param _fname: name of file to save the resulting data to (for the moment, in ASCII format)
    :param _pr: switch specifying if printing tracing the execution should be done or not
    :return: 1D array with (C-aligned) resulting intensity data
    """

    if _pr:
        print("Extracting intensity and saving it to a file ... ", end="")
        t0 = time.time()

    sNumTypeInt = "f"
    if _int_type == 4:
        sNumTypeInt = "d"  # Phase? - if asking for phase, set array to double type

    resMeshI = deepcopy(_wfr.mesh)

    depType = resMeshI.get_dep_type()
    if depType < 0:
        Exception("Incorrect numbers of points in the mesh structure")

    arI = srwlib.array(sNumTypeInt, [0] * resMeshI.ne * resMeshI.nx * resMeshI.ny)
    srwl.CalcIntFromElecField(
        arI,
        _wfr,
        _pol,
        _int_type,
        depType,
        resMeshI.eStart,
        resMeshI.xStart,
        resMeshI.yStart,
    )

    if _det is not None:
        resStkDet = _det.treat_int(arI, resMeshI)
        arI = resStkDet.arS
        resMeshI = resStkDet.mesh

    if len(_fname) > 0:
        srwl_uti_save_intens_ascii(
            arI,
            resMeshI,
            _fname,
            0,
            ["Photon Energy", "Horizontal Position", "Vertical Position", ""],
            _arUnits=["eV", "m", "m", "ph/s/.1%bw/mm^2"],
        )
    if _pr:
        print("completed (lasted", round(time.time() - t0, 2), "s)")

    return arI, resMeshI


# Read and plot generic SRW .dat files created
def read_srw_file(filename):
    data, mode, ranges, labels, units = srw_io.file_load(filename)
    data = np.array(data).reshape((ranges[8], ranges[5]), order="C")
    return {
        "data": data,
        "shape": data.shape,
        "mean": np.mean(data),
        "photon_energy": ranges[0],
        "horizontal_extent": ranges[3:5],
        "vertical_extent": ranges[6:8],
        # 'mode': mode,
        "labels": labels,
        "units": units,
    }


# RMS beam size calculation
def rmsfile(file):
    flux0 = read_srw_file(file)
    data2D = flux0["data"]
    datax = np.sum(data2D, axis=1)
    datay = np.sum(data2D, axis=0)
    hx = flux0["horizontal_extent"]
    hy = flux0["vertical_extent"]
    shape = flux0["shape"]
    xmin = hx[0]
    xmax = hx[1]
    ymin = hy[0]
    ymax = hy[1]
    Nx = shape[0]
    Ny = shape[1]
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    xvals = np.linspace(xmin, xmax, Nx)
    yvals = np.linspace(ymin, ymax, Ny)
    sxsq = sum(datax * xvals * xvals) / sum(datax)
    xavg = sum(datax * xvals) / sum(datax)
    sx = sqrt(sxsq - xavg * xavg)

    sysq = sum(datay * yvals * yvals) / sum(datay)
    yavg = sum(datay * yvals) / sum(datay)
    sy = sqrt(sysq - yavg * yavg)
    return sx, sy


# transform SRW intensity file format to matrix style
def transformSRWIntensityFile(filein, fileout):
    flux0 = read_srw_file(filein)
    data2D = flux0["data"]
    datax = np.sum(data2D, axis=1)
    datay = np.sum(data2D, axis=0)
    hx = flux0["horizontal_extent"]
    hy = flux0["vertical_extent"]
    shape = flux0["shape"]
    xmin = hx[0]
    xmax = hx[1]
    ymin = hy[0]
    ymax = hy[1]
    Nx = shape[0]
    Ny = shape[1]
    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    header1 = "#xmin,xmax,Nx = " + str(xmin) + "," + str(xmax) + "," + str(Nx) + "\n"
    header2 = "#ymin,ymax,Ny=" + str(ymin) + "," + str(ymax) + "," + str(Ny)
    head = header1 + header2
    np.savetxt(fileout, data2D, delimiter=",", header=head, comments="")
    # np.savetxt("results/intx_1mmAperture", datax, delimiter=',', header="Intensity", comments=""


# Polarization from wavefront calculation function
def wfrGetPol(wfr):
    dx = (wfr.mesh.xFin - wfr.mesh.xStart) / wfr.mesh.nx
    dy = (wfr.mesh.yFin - wfr.mesh.yStart) / wfr.mesh.ny
    arReEx = wfr.arEx[::2]
    arImEx = wfr.arEx[1::2]
    arReEy = wfr.arEy[::2]
    arImEy = wfr.arEy[1::2]
    arReEx2d = np.array(arReEx).reshape((wfr.mesh.nx, wfr.mesh.ny), order="C")
    arImEx2d = np.array(arImEx).reshape((wfr.mesh.nx, wfr.mesh.ny), order="C")
    arReEy2d = np.array(arReEy).reshape((wfr.mesh.nx, wfr.mesh.ny), order="C")
    arImEy2d = np.array(arImEy).reshape((wfr.mesh.nx, wfr.mesh.ny), order="C")
    xvals = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx)
    yvals = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny)
    ReEx00 = interpBright(
        0,
        0,
        arReEx2d,
        wfr.mesh.xStart,
        wfr.mesh.yStart,
        dx,
        dy,
        wfr.mesh.nx,
        wfr.mesh.ny,
    )
    ImEx00 = interpBright(
        0,
        0,
        arImEx2d,
        wfr.mesh.xStart,
        wfr.mesh.yStart,
        dx,
        dy,
        wfr.mesh.nx,
        wfr.mesh.ny,
    )
    ReEy00 = interpBright(
        0,
        0,
        arReEy2d,
        wfr.mesh.xStart,
        wfr.mesh.yStart,
        dx,
        dy,
        wfr.mesh.nx,
        wfr.mesh.ny,
    )
    ImEy00 = interpBright(
        0,
        0,
        arImEy2d,
        wfr.mesh.xStart,
        wfr.mesh.yStart,
        dx,
        dy,
        wfr.mesh.nx,
        wfr.mesh.ny,
    )
    norm = math.sqrt(
        ReEx00**2 + ImEx00**2 + ReEy00**2 + ImEy00**2
    )  ##Normalization so that abs(Re[Pvec])^2+abs(Im[Pvec])^2=1
    Pvec = (1 / norm) * np.array([ReEx00 + ImEx00 * (1j), ReEy00 + ImEy00 * (1j)])
    return Pvec


# Intensity from electric fields calculation function
def calc_int_from_elec(_wfr):

    # total real and imag components of electric field
    re0, re0_mesh = calc_int_from_wfr(
        _wfr, _pol=6, _int_type=5, _det=None, _fname="", _pr=False
    )
    im0, im0_mesh = calc_int_from_wfr(
        _wfr, _pol=6, _int_type=6, _det=None, _fname="", _pr=False
    )

    # reshape to 2d mesh
    elec_fields_re = (
        np.array(re0).reshape((_wfr.mesh.ny, -1), order="C").astype(np.float64)
    )
    elec_fields_im = (
        np.array(im0).reshape((_wfr.mesh.ny, -1), order="C").astype(np.float64)
    )

    # calculate intensity
    slice_intensity = (
        0.5
        * const.c
        * const.epsilon_0
        * (elec_fields_re**2.0 + elec_fields_im**2.0)
    )

    return slice_intensity


def extract_2d_fields(_wfr):

    # Extract horizontal component of electric field
    re0_ex, re0_mesh_ex = calc_int_from_wfr(
        _wfr, _pol=0, _int_type=5, _det=None, _fname="", _pr=False
    )
    im0_ex, im0_mesh_ex = calc_int_from_wfr(
        _wfr, _pol=0, _int_type=6, _det=None, _fname="", _pr=False
    )

    # Extract vertical component of electric field
    re0_ey, re0_mesh_ey = calc_int_from_wfr(
        _wfr, _pol=1, _int_type=5, _det=None, _fname="", _pr=False
    )
    im0_ey, im0_mesh_ey = calc_int_from_wfr(
        _wfr, _pol=1, _int_type=6, _det=None, _fname="", _pr=False
    )

    # Reshape arrays from 1d to 2d
    re_ex_2d = (
        np.array(re0_ex)
        .reshape((_wfr.mesh.nx, _wfr.mesh.ny), order="C")
        .astype(np.float64)
    )
    im_ex_2d = (
        np.array(im0_ex)
        .reshape((_wfr.mesh.nx, _wfr.mesh.ny), order="C")
        .astype(np.float64)
    )
    re_ey_2d = (
        np.array(re0_ey)
        .reshape((_wfr.mesh.nx, _wfr.mesh.ny), order="C")
        .astype(np.float64)
    )
    im_ey_2d = (
        np.array(im0_ey)
        .reshape((_wfr.mesh.nx, _wfr.mesh.ny), order="C")
        .astype(np.float64)
    )

    return re_ex_2d, im_ex_2d, re_ey_2d, im_ey_2d


def make_wavefront(ex_re_2d, ex_im_2d, ey_re_2d, ey_im_2d, photon_e_ev, x, y):

    # Flatten fields
    re_ex = ex_re_2d.flatten(order="C")
    im_ex = ex_im_2d.flatten(order="C")
    re_ey = ey_re_2d.flatten(order="C")
    im_ey = ey_im_2d.flatten(order="C")

    # Combine real and imaginary fields into srw-preferred format
    ex_numpy = np.zeros(2 * len(re_ex))
    for i in range(len(re_ex)):
        ex_numpy[2 * i] = re_ex[i]
        ex_numpy[2 * i + 1] = im_ex[i]

    ey_numpy = np.zeros(2 * len(re_ey))
    for i in range(len(re_ey)):
        ey_numpy[2 * i] = re_ey[i]
        ey_numpy[2 * i + 1] = im_ey[i]

    # Convert to list
    ex = array("f", ex_numpy.tolist())
    ey = array("f", ey_numpy.tolist())

    # Pass changes to SRW
    wfr1 = srwlib.SRWLWfr(
        _arEx=ex,
        _arEy=ey,
        _typeE="f",
        _eStart=photon_e_ev,
        _eFin=photon_e_ev,
        _ne=1,
        _xStart=np.min(x),
        _xFin=np.max(x),
        _nx=len(x),
        _yStart=np.min(y),
        _yFin=np.max(y),
        _ny=len(y),
        _zStart=0.0,
        _partBeam=None,
    )
    return wfr1
