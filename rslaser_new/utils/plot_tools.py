# -*- coding: utf-8 -*-
"""Plotting methods specific to laser pulses
Copyright (c) 2021-2022 RadiaSoft LLC. All rights reserved
"""
import math
from matplotlib.path import Path
import numpy as np


def scatter_contour(plot_flag, plot_type, x, y, ax, divs=10, levels=10):
    """Generalized algorithm for plotting contour and/or scatter plots.

    Adapted from open source method: scatter_contour.py
    https://github.com/astroML/astroML/blob/master/astroML/plotting/scatter_contour.py

    Args:
        plot_flag (string): style of plot (scatter, contour, line, etc.)
        plot_type (string): axis scaling (linear, log-log, or semi-log)
        x, y (2d array): x and y data for the contour plot
        ax (axis obj): the axes on which to plot
        divs (int): desired number of divisions along each axis (int)
        levels (int *or* array): number of contour levels *or* an array of levels
    """
    ref = None
    if plot_flag in ["contour", "combo"]:
        if type(x) is list:  # x contains data for 2 axis ranges
            levels = np.asarray(levels)
            if levels.size == 1:
                levels = np.linspace(min(y), max(y), levels)

            minX = min(x[0])
            maxX = max(x[0])

            minY = min(x[1])
            maxY = max(x[1])

            points = len(y)
            ratio = float(maxX - minX) / (maxY - minY)
            shapeX = math.sqrt(points * ratio)
            shapeY = math.sqrt(points / ratio)
            X, Y = np.meshgrid(x[0], x[1])
            Z = y.reshape([shapeX, shapeY])
            Z = Z[0 : len(x[0]), 0 : len(x[1])]

            ref = ax.contourf(X, Y, Z, levels=levels, extent=[minX, maxX, minY, maxY])

        else:
            threshold = 8 if plot_flag == "combo" else 1

            # generate the 2D histogram, allowing the algorithm to use
            #   all data points, automatically calculating the 2D extent
            myHist, xbins, ybins = np.histogram2d(x, y, divs)

            # specify contour levels, allowing user to input simple integer
            levels = np.asarray(levels)
            # if user specified an integer, then populate levels reasonably
            if levels.size == 1:
                levels = np.linspace(threshold, myHist.max(), levels)

            # define the 'extent' of the contoured area, using the
            #   the horizontal and vertical arrays generaed by histogram2d()
            extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]]
            i_min = np.argmin(levels)

            # draw a zero-width line, which defines the outer polygon,
            #   in order to reduce the number of points drawn
            outline = ax.contour(
                myHist.T, levels[i_min : i_min + 1], linewidths=0, extent=extent
            )

            # generate the contoured image, filled or not
            #   use myHist.T, rather than full myHist, to limit extent of the contoured region
            #   i.e. only the high-density regions are contoured
            #   the return value is potentially useful to the calling method
            ref = ax.contourf(myHist.T, levels, extent=extent)

    # logic for finding particles in low-density regions
    if plot_flag == "combo":
        # create new 2D array that will hold a subset of the particles
        #   i.e. only those in the low-density regions
        lowDensityArray = np.hstack([x[:, None], y[:, None]])

        # extract only those particles outside the high-density region
        if len(outline.allsegs[0]) > 0:
            outer_poly = outline.allsegs[0][0]
            points_inside = Path(outer_poly).contains_points(lowDensityArray)
            Xplot = lowDensityArray[~points_inside]
        else:
            Xplot = lowDensityArray

    if plot_flag.startswith("scatter") or plot_flag.endswith("line"):
        Xplot = np.hstack([x[:, None], y[:, None]])

    if plot_flag in ["combo", "scatter", "scatter-line"]:

        # Terrible hack to get around the "fact" that scatter plots
        # do not get correct axis limits if either axis is log scale.
        # ax.plot(...) seems to work, so draw a plot and then delete
        # it, leaving the plot with a correct axes view.

        (toRemove,) = ax.plot(Xplot[:, 0], Xplot[:, 1], c="w")
        ax.scatter(Xplot[:, 0], Xplot[:, 1], marker=".", c="k")
        ax.lines.remove(toRemove)

    if plot_flag.endswith("line"):
        ax.plot(Xplot[:, 0], Xplot[:, 1], c="k")

    if plot_flag in ["line", "scatter", "scatter-line"]:
        if plot_type in ["log-log", "semi-logx"]:
            ax.set_xscale("log", nonposx="mask")

        if plot_type in ["log-log", "semi-logy"]:
            ax.set_yscale("log", nonposy="mask")

        if plot_type in ["linear", "semi-logy"]:
            ax.set_xscale("linear")

        if plot_type in ["linear", "semi-logx"]:
            ax.set_yscale("linear")

    return ref


def generate_contour_levels(field, n_levels=40, multiplier=1.1):
    """Generate the contour levels.

    Args:
        field (2d array): array of values to be contoured
        n_levels (int): number of contour levels
        multiplier (float): should be just slightly larger than 1.0

    Returns:
        f_levels:  list of values between min/max of argument 'field'
    """
    multiplier_alt = 2.0 - multiplier  # flips 1.1 into 0.9

    # slightly increase the max and decrease the min
    if f_max > 0.0:
        f_max = multiplier * np.max(field)
    else:
        f_max = multiplier_alt * np.max(field)

    if f_min < 0.0:
        f_min = multiplier * np.min(field)
    else:
        f_min = multiplier_alt * np.min(field)

    # generate symmetric min/max values
    if abs(f_min) < f_max:
        f_max = np.around(f_max, decimals=3)
        f_min = -f_max
    else:
        f_min = np.around(f_min, decimals=3)
        f_max = abs(f_min)

    # create the level values
    f_levels = []
    delta_f = (f_max - f_min) / n_levels
    for i in range(n_levels):
        f_levels.append(f_min + i * delta_f)

    return f_levels


def round_sig_fig(value, n_digits):
    """Round floating point value to specified number of significant figures.

    Args:
        value (float): the floating point number to be rounded
        n_digits (int): number of significant figures

    Returns:
        the rounded floating point value; returns 0 in case of error
    """
    try:
        # find a, b such that value = a*10^b (1 <= a < 10)
        b = math.floor(math.log10(abs(value)))
        a = value / 10**b
        return round(a, n_digits - 1) * 10**b
    except ValueError:
        return 0


#
def print_nd_message(plot_flag, plot_dimension):
    """Jupyter notebook helper; tell user that 2D or 3D plots are not being rendered.

    Args:
        plot_flag (bool): indicates whether plot will be generated in the notebook cell
        plot_dimension (int): either 2 or 3
    """
    # if plot is not to be rendered, then print the message
    if plot_flag == False:
        print(" ")
        print("********************************")
        print(str(plot_dimension) + "D plots are not being rendered.")
        print("********************************")
        print(" ")
