'''
Various utilities for correction and conversion of satellite data.

@title: correct
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 02/03/2020
    correct_parallax
    compute_gaussian_curvature
'''
import numpy as np


def correct_parallax(sr_x, sr_y, gpmset):
    '''
    Adjust the geo-locations of the SR pixels.
    The `sr_xy` coordinates of the SR beam footprints need to be in the
    azimuthal equidistant projection of the ground radar. This ensures that the
    ground radar is fixed at xy-coordinate (0, 0), and every SR bin has its
    relative xy-coordinates with respect to the ground radar site.

    Parameters
    ----------
    sr_xy : :class:`numpy:numpy.ndarray`
        Array of xy-coordinates of shape (nscans, nbeams, 2)
    gpmset: xarray

    Returns
    -------
    sr_xyp : :class:`numpy:numpy.ndarray`
        Array of parallax corrected coordinates
        of shape (nscans, nbeams, nbins).
    z_sr : :class:`numpy:numpy.ndarray`
        Array of SR bin altitudes of shape (nscans, nbeams, nbins).
    '''
    r_sr_inv, alpha = gpmset.nbin.values, gpmset.nray.values
    # calculate height of bin
    z = r_sr_inv * np.cos(np.deg2rad(alpha))[..., np.newaxis]
    z_sr = np.repeat(z[np.newaxis, :, :], len(gpmset.nscan), axis=0)
    # calculate bin ground xy-displacement length
    ds = r_sr_inv * np.sin(np.deg2rad(alpha))[..., np.newaxis]

    # calculate x,y-differences between ground coordinate
    # and center ground coordinate [25th element]
    center = int(np.floor(len(sr_x[-1]) / 2.0))
    xdiff = sr_x - sr_x[:, center][:, np.newaxis]
    ydiff = sr_y - sr_y[:, center][:, np.newaxis]

    # assuming ydiff and xdiff being a triangles adjacent and
    # opposite this calculates the xy-angle of the SR scan
    ang = np.arctan2(ydiff, xdiff)

    # calculate displacement dx, dy from displacement length
    dx = ds * np.cos(ang)[..., np.newaxis]
    dy = ds * np.sin(ang)[..., np.newaxis]

    # subtract displacement from SR ground coordinates
    sr_xp = sr_x[..., np.newaxis] - dx
    sr_yp = sr_y[..., np.newaxis] - dy

    return sr_xp, sr_yp, z_sr


def compute_gaussian_curvature(lat0):
    '''
    Determine the Earth's Gaussian radius of curvature at the radar
    https://en.wikipedia.org/wiki/Earth_radius#Radii_of_curvature

    Parameter:
    ----------
    lat0: float
        Ground radar latitude.

    Returns:
    --------
    ae: float
        Earth's Gaussian radius.
    '''
    # Major and minor radii of the Ellipsoid
    a = 6378137.0  # Earth radius in meters
    e2 = 0.0066943800
    b = a * np.sqrt(1 - e2)

    tmp = (a * np.cos(np.pi / 180 * lat0))**2 + (b * np.sin(np.pi / 180 * lat0))**2   # Denominator
    an = (a**2) / np.sqrt(tmp)  # Radius of curvature in the prime vertical (east–west direction)
    am = (a * b)**2 / tmp ** 1.5  # Radius of curvature in the north–south meridian
    ag = np.sqrt(an * am)  # Earth's Gaussian radius of curvature
    ae = (4 / 3.) * ag

    return ae
