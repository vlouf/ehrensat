'''
Utilities to read the input data and format them in a way to be read by
volume_matching.
@title: ehrensat
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 02/03/2020
    get_gpm_orbit
    read_GPM
'''
import re
import datetime

import h5py
import numpy as np
import xarray as xr


def get_gpm_orbit(gpmfile):
    '''
    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    Returns:
    --------
    orbit: int
        GPM Granule Number.
    '''
    try:
        with h5py.File(gpmfile) as hid:
            grannb = [s for s in hid.attrs['FileHeader'].split() if b'GranuleNumber' in s][0].decode('utf-8')
            orbit = re.findall('[0-9]{3,}', grannb)[0]
    except Exception:
        return 0

    return int(orbit)


def read_GPM(infile, refl_min_thld):
    '''
    Read GPM data and organize them into a Dataset.
    Parameters:
    ----------
    gpmfile: str
        GPM data file.
    refl_min_thld: float
        Minimum threshold applied to GPM reflectivity.
    Returns:
    --------
    dset: xr.Dataset
        GPM dataset.
    '''
    data = dict()
    date = dict()
    with h5py.File(infile, 'r') as hid:
        keys = hid['/NS'].keys()
        for k in keys:
            if k == 'Latitude' or k == 'Longitude':
                dims = tuple(hid[f'/NS/{k}'].attrs['DimensionNames'].decode('UTF-8').split(','))
                fv =  hid[f'/NS/{k}'].attrs['_FillValue']
                data[k] = (dims, np.ma.masked_equal(hid[f'/NS/{k}'][:], fv))
            else:
                subkeys = hid[f'/NS/{k}'].keys()
                for sk in subkeys:
                    dims = tuple(hid[f'/NS/{k}/{sk}'].attrs['DimensionNames'].decode('UTF-8').split(','))
                    fv =  hid[f'/NS/{k}/{sk}'].attrs['_FillValue']

                    if sk in ['Year', 'Month', 'DayOfMonth', 'Hour', 'Minute', 'Second', 'MilliSecond']:
                        date[sk] = np.ma.masked_equal(hid[f'/NS/{k}/{sk}'][:], fv)
                    elif sk in ['DayOfYear', 'SecondOfDay']:
                        continue
                    elif sk == 'typePrecip':
                        # Simplify precipitation type
                        data[sk] = (dims, hid[f'/NS/{k}/{sk}'][:] / 10000000)
                    elif sk == 'zFactorCorrected':
                        # Reverse direction along the beam.
                        data[sk] = (dims, np.ma.masked_less_equal(hid[f'/NS/{k}/{sk}'][:][:, :, ::-1], refl_min_thld))
                    else:
                        data[sk] = (dims, np.ma.masked_equal(hid[f'/NS/{k}/{sk}'][:], fv))

    try:
        data['zFactorCorrected']
    except Exception:
        raise KeyError(f"GPM Reflectivity not found in {infile}")

    # Create Quality indicator.
    quality = np.zeros(data['heightBB'][-1].shape, dtype=np.int32)
    quality[((data['qualityBB'][-1] == 0) | (data['qualityBB'][-1] == 1)) & (data['qualityTypePrecip'][-1] == 1)] = 1
    quality[(data['qualityBB'][-1] > 1) | (data['qualityTypePrecip'][-1] > 1)] = 2
    data['quality'] = (data['heightBB'][0], quality)

    # Generate dimensions.
    nray = np.linspace(-17.04, 17.04, 49)
    nbin = np.arange(0, 125 * 176, 125)

    R, A = np.meshgrid(nbin, nray)
    distance_from_sr = 407000 / np.cos(np.deg2rad(A)) - R  # called rt in IDL code.
    data['distance_from_sr'] = (('nray', 'nbin'), distance_from_sr)

    dtime = np.array([datetime.datetime(*d) for d in zip(date['Year'],
                                                         date['Month'],
                                                         date['DayOfMonth'],
                                                         date['Hour'],
                                                         date['Minute'],
                                                         date['Second'],
                                                         date['MilliSecond'])], dtype='datetime64')

    data['nscan'] = (('nscan'), dtime)
    data['nray'] = (('nray'), nray)
    data['nbin'] = (('nbin'), nbin)

    dset = xr.Dataset(data)

    dset.nray.attrs = {'units': 'degree', 'description':'Deviation from Nadir'}
    dset.nbin.attrs = {'units': 'm', 'description':'Downward from 0: TOA to Earth ellipsoid.'}
    dset.attrs['altitude'] = 407000
    dset.attrs['altitude_units'] = 'm'
    dset.attrs['altitude_description'] = "GPM orbit"
    dset.attrs['beamwidth'] = 0.71
    dset.attrs['beamwidth_units'] = 'degree'
    dset.attrs['beamwidth_description'] = "GPM beamwidth"
    dset.attrs['dr'] = 125
    dset.attrs['dr_units'] = 'm'
    dset.attrs['dr_description'] = "GPM gate spacing"
    dset.attrs['orbit'] = get_gpm_orbit(infile)

    return dset
