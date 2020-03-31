'''
Utilities to read the input data and format them in a way to be read by
volume_matching.
@title: ehrensat
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 31/03/2020
    get_overpass_with_precip
'''
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd

import dask
import dask.bag as db
from dask.diagnostics import ProgressBar

import ehrensat
from ehrensat.ehrensat import NoPrecipitationError


def get_overpass_with_precip(gpmfile, radarset):
    gpmset = ehrensat.read_GPM(gpmfile)
    data = dict()
    for n in range(len(radarset)):
        rid = radarset.id[n]
        rname = radarset.short_name[n]
        grlat = radarset.site_lat[n]
        grlon = radarset.site_lon[n]

        try:
            nprof, gpmtime = ehrensat.precip_in_domain(gpmset, grlat=grlat, grlon=grlon)
        except NoPrecipitationError:
            continue
        # print(f'{rid} - {rname} radar at ({grlon}, {grlat}) has {nprof} matches with GPM at {gpmtime}.')

        data[rid] = (str(gpmtime),
                     rname,
                     str(grlon),
                     str(grlat),
                     str(nprof),
                     gpmfile)
    return data


def savedata(rslt):
    for n in rslt:
        for rid in n.keys():
            outpath = '/scratch/kl02/vhl548/gpm_output/overpass'
            outfile = os.path.join(outpath, f'gpm.{rid:02}.csv')
            with open(outfile, 'a+') as fid:
                fid.write(','.join(n[rid]))
                fid.write('\n')
    return None


def main():
    flist = np.array([])
    for year in [2018, 2019, 2020]:
        flist = np.append(flist, sorted(glob.glob(f'/g/data/rq0/admin/calibration/sr_data/gpm_data/{year}/**/**/*.*')))

    df = pd.read_csv('./radar_site_list.csv')
    ndf = df.drop_duplicates('id', keep='last').reset_index()
    argslist = [(f, ndf) for f in flist]

    bag = db.from_sequence(argslist).starmap(get_overpass_with_precip)
    with ProgressBar():
        rslt = bag.compute()

    savedata(rslt)

    return None


if __name__ == "__main__":
    main()
