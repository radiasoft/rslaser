# -*- coding: utf-8 -*-
"""Methods for plotting electromagnetic fields.
Copyright (c) 2021 RadiaSoft LLC. All rights reserved
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pykern.pkdebug import pkdc
import numpy as np

import rslaser_new.utils.plot_tools as rspt


def plot_1d_x(_xArr, _pulse, _ax, _y=0.0, _z=0.0, _t=0.0, _time_explicit=False):
    """Plot a 1D (transverse, along x) lineout of a laser pulse field.

    For now, we are assuming the field is Ex

    Args:
        _xArr (1D numpy array): x positions where the field is to be evaluated
        _pulse (object): instance of the LaserPulseEnvelope() class.
        _ax (object): matplotlib 'axis', used to generate the plot
        _y (float): [m] vertical location of the lineout
        _z (float): [m] longitudinal location of the lineout
        _t (float): [m] time of the lineout
        _time_explicit (bool): envelope only (False) or time explicit (True)
    """
    numX = np.size(_xArr)

    # Calculate Ex
    em_field = np.zeros(numX)
    if _time_explicit:
        em_field = np.real(_pulse.evaluate_ex(_xArr, _y, _z, _t))
    else:
        em_field = np.real(_pulse.evaluate_envelope_ex(_xArr, _y, _z))

    _ax.plot(_xArr, em_field)
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("Ex [V/m]")
    if _time_explicit:
        _ax.set_title(
            "Ex [V/m], at (y,z)=({0:4.2f},{0:4.2f}) [m] and t={0:4.2f} [s]".format(
                _y, _z, _t
            )
        )
    else:
        _ax.set_title(
            "Ex (envelope) [V/m], at (y,z)=({0:4.2f},{0:4.2f}) [m]".format(_y, _z)
        )


def plot_1d_y(_yArr, _pulse, _ax, _x=0.0, _z=0.0, _t=0.0, _time_explicit=False):
    """Plot a 1D (transverse, along y) lineout of a laser pulse field.

    For now, we are assuming the field is Ex

    Args:
        _yArr (1D numpy array): y positions where the field is to be evaluated
        _pulse (object): instance of the LaserPulseEnvelope() class.
        _ax (object): matplotlib 'axis', used to generate the plot
        _x (float): [m] horizontal location of the lineout
        _z (float): [m] longitudinal location of the lineout
        _t (float): [m] time of the lineout
        _time_explicit (bool): envelope only (False) or time explicit (True)
    """
    numY = np.size(_yArr)

    # Calculate Ex
    em_field = np.zeros(numY)
    if _time_explicit:
        em_field = np.real(_pulse.evaluate_ex(_x, _yArr, _z, _t))
    else:
        em_field = np.real(_pulse.evaluate_envelope_ex(_x, _yArr, _z))

    _ax.plot(_yArr, em_field)
    _ax.set_xlabel("y [m]")
    _ax.set_ylabel("Ex [V/m]")
    if _time_explicit:
        _ax.set_title(
            "Ex [V/m], at (x,z)=({0:4.2f},{0:4.2f}) [m] and t={0:4.2f} [s]".format(
                _x, _z, _t
            )
        )
    else:
        _ax.set_title(
            "Ex (envelope) [V/m], at (x,z)=({0:4.2f},{0:4.2f}) [m]".format(_x, _z)
        )


def plot_1d_r(_rArr, _pulse, _ax, _z=0.0, _t=0.0, _time_explicit=False):
    """Plot a 1D (transverse, along r) lineout of a laser pulse field.

    For now, we are assuming circular polarization.

    Args:
        _rArr (1D numpy array): r positions where the field is to be evaluated
        _pulse (object): instance of the LaserPulseEnvelope() class.
        _ax (object): matplotlib 'axis', used to generate the plot
        _z (float): [m] longitudinal location of the lineout
        _t (float): [m] time of the lineout
        _time_explicit (bool): envelope only (False) or time explicit (True)
    """
    numR = np.size(_rArr)

    # Calculate Ex
    em_field = np.zeros(numR)
    if _time_explicit:
        em_field = np.real(_pulse.evaluate_er(_rArr, _z, _t))
    else:
        em_field = np.real(_pulse.evaluate_envelope_er(_rArr, _z))

    _ax.plot(_rArr, em_field)
    _ax.set_xlabel("r [m]")
    _ax.set_ylabel("Er [V/m]")
    if _time_explicit:
        _ax.set_title("Er [V/m], at z={0:4.2f} [m] and t={0:4.2f} [s]".format(_z, _t))
    else:
        _ax.set_title("Er (envelope) [V/m], at z={0:4.2f} [m]".format(_z))


def plot_1d_z(_zArr, _pulse, _ax, _x=0.0, _y=0.0, _t=0.0, _time_explicit=False):
    """Plot a 1D (longitudinal, along z) lineout of a laser pulse field.

    For now, we are assuming the field is Ex

    Args:
        _zArr (1D numpy array): z positions where the field is to be evaluated
        _pulse (object): instance of the LaserPulseEnvelope() class.
        _ax (matplotlib axis): used to generate plot
        _x (float): [m] horizontal location of the lineout
        _y (float): [m] vertical location of the lineout
        _t (float): [m] time of the lineout
        _time_explicit (bool): envelope only (False) or time explicit (True)
    """
    numZ = np.size(_zArr)

    # Calculate Ex at the 2D array of x,y values
    em_field = np.zeros(numZ)
    if _time_explicit:
        for iLoop in range(numZ):
            em_field[iLoop] = np.real(_pulse.evaluate_ex(_x, _y, _zArr[iLoop], _t))
    else:
        for iLoop in range(numZ):
            em_field[iLoop] = np.real(_pulse.evaluate_envelope_ex(_x, _y, _zArr[iLoop]))

    _ax.plot(_zArr, em_field)
    _ax.set_xlabel("z [m]")
    _ax.set_ylabel("Ex [V/m]")
    if _time_explicit:
        _ax.set_title(
            "Ex [V/m], at (x,y)=({0:4.2f},{0:4.2f}) [m] and t={0:4.2f} [s]".format(
                _x, _y, _t
            )
        )
    else:
        _ax.set_title(
            "Ex (envelope) [V/m], at (x,y)=({0:4.2f},{0:4.2f}) [m]".format(_x, _y)
        )


def plot_2d_zy(
    _zArr, _yArr, _pulse, _ax, _x=0.0, _t=0.0, _time_explicit=False, _nlevels=40
):

    numZ = np.size(_zArr)
    numY = np.size(_yArr)

    # Calculate Ex at the 2D array of z,y values
    #    zyEData = np.zeros((numZ, numY))
    zyEData = np.zeros((numY, numZ))
    if _time_explicit:
        for i in range(numZ):
            zyEData[:, i] = np.real(_pulse.evaluate_ex(_x, _yArr, _zArr[i], _t))
    else:
        for i in range(numZ):
            zyEData[:, i] = np.real(_pulse.evaluate_envelope_ex(_x, _yArr, _zArr[i]))

    # generate the contour plot
    _ax.clear()
    _ax.axis([_zArr.min(), _zArr.max(), _yArr.min(), _yArr.max()])
    _ax.set_xlabel("z [m]")
    _ax.set_ylabel("y [m]")
    _ax.set_title("ZY slice, at  x={0:4.2f} [m]".format(_x))

    if _time_explicit:
        del_level = rspt.round_sig_fig(0.202 * zyEData.max(), 3)
        n_cbar_labels = 10  # choose an even number
        max_level = n_cbar_labels * del_level / 2
        _levels = np.linspace(-max_level, max_level, _nlevels)
    else:
        del_level = rspt.round_sig_fig(0.101 * zyEData.max(), 3)
        n_cbar_labels = 10  # choose an even number
        max_level = n_cbar_labels * del_level
        _levels = np.linspace(0.0, max_level, _nlevels)

    contours = _ax.contourf(_zArr, _yArr, zyEData, _levels, extent="none")

    # generate the colorbar
    divider = make_axes_locatable(_ax)
    _cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(contours, format="%3.2e", cax=_cax)
    tick_values = []
    if _time_explicit:
        for i in range(n_cbar_labels + 1):
            tick_values.append(-max_level + i * del_level)
    else:
        for i in range(n_cbar_labels + 1):
            tick_values.append(i * del_level)
    cbar.set_ticks(tick_values)


def plot_2d_xy(
    _xArr, _yArr, _pulse, _ax, _z=0.0, _t=0.0, _time_explicit=False, _nlevels=40
):
    """Generate a 2D contour plot of Ex in the transverse plane.

    For now, we are assuming the field is Ex

    Args:
        _xArr (1D numpy array): x positions where the field is to be evaluated
        _yArr (1D numpy array): x positions where the field is to be evaluated
        _pulse (object): instance of the LaserPulseEnvelope() class.
        _ax (object): matplotlib 'axis', used to generate the plot
        _z (float): [m] longitudinal location of the lineout
        _t (float): [m] time of the lineout
        _time_explicit (bool): envelope only (False) or time explicit (True)
    """
    numX = np.size(_xArr)
    numY = np.size(_yArr)

    # Calculate Ex at the 2D array of x,y values
    xyEData = np.zeros((numX, numY))
    for iLoop in range(numY):
        xyEData[iLoop, :] = np.real(
            _pulse.evaluate_envelope_ex(_xArr[iLoop], _yArr, _z)
        )

    # manually set the plot limits
    xMin = np.min(_xArr)
    xMax = np.max(_xArr)
    yMin = np.min(_yArr)
    yMax = np.max(_yArr)

    # generate the contour plot
    _ax.clear()
    _ax.axis([xMin, xMax, yMin, yMax])
    _ax.axis("equal")
    _ax.set_xlabel("x [m]")
    _ax.set_ylabel("y [m]")
    if _time_explicit:
        _ax.set_title("Ex [V/m], at z=({0:4.2f} [m] and t={0:4.2f} [s]".format(_z, _t))
    else:
        _ax.set_title("Ex (envelope) [V/m], at z={0:4.2f} [m]".format(_z))

    n_cbar_labels = 10
    min_level = rspt.round_sig_fig(1.01 * min(xyEData.min(), 0.0), 3)
    max_level = rspt.round_sig_fig(1.01 * xyEData.max(), 3)
    del_level = rspt.round_sig_fig((max_level - min_level) / n_cbar_labels, 3)

    _levels = np.linspace(min_level, max_level, _nlevels)
    contours = _ax.contourf(_xArr, _yArr, xyEData, _levels, extent="none")

    # generate the colorbar
    divider = make_axes_locatable(_ax)
    _cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(contours, format="%3.2e", cax=_cax)
    tick_values = []
    for i in range(n_cbar_labels + 1):
        tick_values.append(min_level + i * del_level)
    cbar.set_ticks(tick_values)


def plot_2d_zr(
    _zArr, _rArr, _pulse, _ax, _x=0.0, _t=0.0, _time_explicit=False, _nlevels=40
):

    numZ = np.size(_zArr)
    numR = np.size(_rArr)

    # Calculate Ex at the 2D array of z,r values
    zrEData = np.zeros((numR, numZ))
    if _time_explicit:
        for i in range(numZ):
            for j in range(numR):
                zrEData[j, i] = np.real(_pulse.evaluate_er(_rArr[j], _zArr[i], _t))
    else:
        for i in range(numZ):
            zrEData[:, i] = np.real(_pulse.evaluate_envelope_er(_rArr, _zArr[i]))

    # generate the contour plot
    _ax.clear()
    _ax.axis([_zArr.min(), _zArr.max(), 0.0, _rArr.max()])
    _ax.set_xlabel("z [m]")
    _ax.set_ylabel("r [m]")
    _ax.set_title("ZY slice, at  x={0:4.2f} [m]".format(_x))

    if _time_explicit:
        del_level = rspt.round_sig_fig(0.202 * zrEData.max(), 3)
        n_cbar_labels = 10  # choose an even number
        max_level = n_cbar_labels * del_level / 2
        _levels = np.linspace(-max_level, max_level, _nlevels)
    else:
        del_level = rspt.round_sig_fig(0.101 * zrEData.max(), 3)
        n_cbar_labels = 10  # choose an even number
        max_level = n_cbar_labels * del_level
        _levels = np.linspace(0.0, max_level, _nlevels)

    contours = _ax.contourf(_zArr, _rArr, zrEData, _levels, extent="none")

    # generate the colorbar
    divider = make_axes_locatable(_ax)
    _cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(contours, format="%3.2e", cax=_cax)
    tick_values = []
    if _time_explicit:
        for i in range(n_cbar_labels + 1):
            tick_values.append(-max_level + i * del_level)
    else:
        for i in range(n_cbar_labels + 1):
            tick_values.append(i * del_level)
    cbar.set_ticks(tick_values)
