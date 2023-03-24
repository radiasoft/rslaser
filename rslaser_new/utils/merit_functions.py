# -*- coding: utf-8 -*-
u"""Merit functions used for nonlinear optimization
Copyright (c) 2022 RadiaSoft LLC. All rights reserved
"""
import numpy as np

def spline_wfs(x, y, xc, yc, r_mid, r_max, phi_min, phi_max, r0):
    """Crude spline-based function for fitting wavefront sensor (WFS) data.

    Args:
        x, y (2d array): horizontal and vertical positions for the wavefront phase data
        xc, yc (float): approximate center of the laser pulse
        r_mid (float): fcn is quadratic for r<r_mid, linear for r>r_mid
        r_max (float): approximate maximum value of the radius
        phi_min (float): minimum value of the wavefront phase
        phi_max (float): maximum value of the wavefront phase
        r0 (float): defines quadratic behavior for r<r_mid

    Returns:
        function values (2d array):  phase values of complex E-field in an electromagnetic wavefront
    """
    # calculate radius r, as well as r**2
    rsq = (x-xc)**2 + (y-yc)**2
    r = np.sqrt(rsq)
    
    # two parameters that are uniquely determined by the input args
    phi_mid = phi_max * (1. - (r_mid/r0)**2)
    c0 = (phi_mid - phi_min) / (1. - r_mid/r_max)
 
    # evaluate the function
    f_1 = phi_max * (1. - rsq/r0**2)     # fcn value for r<r_mid
    f_2 = phi_min + c0 * (1. - r/r_max)  # fcn value for r>r_mid
    return np.where(r<=r_mid, f_1, f_2)

def spline_wfs_fit(params, x, y, data, xc, yc, r_mid, r_max, phi_min):
    """Merit function for least-squares fit of crude spline-like function to wavefront sensor (WFS) data.

    Args:
        params[0] (float): defines the quadratic behavior for r<r_mid (fitting parameter)
        params[1] (float): defines the maximum value of the wavefront phase (fitting parameter)
        x, y (2d array): horizontal and vertical positions for the wavefront phase data
        data (2d array): phase values of complex E-field (raw data from experimental diagnostic)
        xc, yc (float): approximate center of the laser pulse
        r_mid (float): least squares fitting is applied to norm(fit-data) for r<r_mid
        r_max (float): approximate maximum value of the radius
        phi_min (float): minimum value of the wavefront phase

    Returns:
        norm of the difference between the fitted array and raw data, only for r<r_mid
    """
    r0 = params[0]
    phi_max = params[1]
    
    # create the fitted array, using input args
    g = spline_wfs(x, y, xc, yc, r_mid, r_max, phi_min, phi_max, r0)
    
    # calculate the radius and use this to weight the difference towards small r values
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    weighted_diff = np.where(r<=r_mid, (g - data.reshape(g.shape))/r, 0.)
    
    # return the norm of the difference between the fitted array and the raw data
    return np.linalg.norm(weighted_diff)

def gaussian_ccd(x, y, xc, yc, n0_max, r_rms):
    """Radially symmetric Gaussian function for fitting wavefront intensity data from a CCD diagnostic.

    Args:
        x, y (2d array): horizontal and vertical positions for the wavefront phase data
        xc, yc (float): approximate center of the laser pulse
        n0_max (float): maximum photon count from the CCD diagnostic
        r_rms (float): RMS radius of the Gaussian distribution

    Returns:
        function values (2d array):  intensity of an electromagnetic wavefront
    """
    return n0_max * np.exp(-(((x-xc)/r_rms)**2 + ((y-yc)/r_rms)**2))

def gaussian_ccd_fit(params, x, y, xc, yc, data, r_mid):
    """Merit function for least-squares fit of radially symmetric Gaussian to wavefront intensity.

    Args:
        params[0] (float): defines the RMS radius of the Gaussian distribution (fitting parameter)
        params[1] (float): defines the maximum photon count from the CCD diagnostic (fitting parameter)
        x, y (2d array): horizontal and vertical positions for the wavefront phase data
        xc, yc (float): approximate center of the laser pulse
        data (2d array): photon counts from CCD camera (raw data from experimental diagnostic)
        r_mid (float): least squares fitting is applied to norm(fit-data) for r<r_mid

    Returns:
        norm of the difference between the fitted array and raw data, only for r<r_mid
    """
    r_rms = params[0]
    n0_max = params[1]

    # create the fitted array, using input args
    g = gaussian_ccd(x, y, xc, yc, n0_max, r_rms)
    
    # calculate the radius and use this to weight the difference towards small r values
    r = np.sqrt((x-xc)**2 + (y-yc)**2)
    weighted_diff = np.where(r<r_mid, (g - data.reshape(g.shape))/r, 0.)

    # return the norm of the difference between the fitted array and the raw data
    return np.linalg.norm(weighted_diff)

# compute an azimuthally averaged radial profile
# Code adapted from https://github.com/vicbonj/radialprofile/blob/master/radialProfile.py

def azimuthalAverage(image, centerx, centery, type='mean'):
    """Compute an azimuthally symmetric profile around the specified center of 2d Cartesian data.

    Args:
        image (2d array): the 2d Cartesian data, for which a radial profile is desired
        centerx, centery (float): approximate center of the data
        type (string): 'mean', 'median' or 'mode' to vary the applied algorithm

    Returns:
        profiles, errors, distance to the center in pixels
    """
    y, x = np.indices(image.shape)
    r = np.hypot(x - centerx, y - centery)

    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    r_int = r_sorted.astype(int)

    deltar = r_int[1:] - r_int[:-1]
    rind = np.where(deltar)[0]
    rind2 = rind+1
    rind3 = np.zeros(len(rind2)+1)
    rind3[1:] = rind2
    rind3 = rind3.astype('int')

    if type == 'mean':
        aaa = [np.nanmean(i_sorted[rind3[i]:rind3[i+1]]) for i in range(len(rind3)-1)]
    elif type == 'median':
        aaa = [np.nanmedian(i_sorted[rind3[i]:rind3[i+1]]) for i in range(len(rind3)-1)]
    elif type == 'mode':
        aaa_list = [i_sorted[rind3[i]:rind3[i+1]] for i in range(len(rind3)-1)]
        aaa = []
        for part in aaa_list:
            if len(part) == 1:
                counts, xed = np.histogram(part, bins=len(part))
            elif (len(part) > 1) & (len(part) < 40):
                counts, xed = np.histogram(part, bins=int(len(part)/2))
            else:
                counts, xed = np.histogram(part, bins=20)
            if len(xed) != 2:
                aaa.append(0.5*(xed[1:]+xed[:-1])[np.where(counts == np.max(counts))[0]][0])
            elif len(xed) == 2:
                aaa.append(0.5*(xed[1:]+xed[:-1])[np.where(counts == np.max(counts))[0]][0])
    else:
        raise ValueError('Nope')
    aaa_std = [np.nanstd(i_sorted[rind3[i]:rind3[i+1]]) for i in range(len(rind3)-1)]
    dist_r = [np.mean(r_sorted[rind3[i]:rind3[i+1]]) for i in range(len(rind3)-1)]
    return np.array(aaa), np.array(aaa_std), np.array(dist_r)
