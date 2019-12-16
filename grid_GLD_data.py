import subprocess
from partition import partition 

import numpy as np
import pandas as pd
import pickle
import gzip

from index_helpers import load_Kp
import matplotlib.pyplot as plt
import os
import itertools
import random
import os
import time
import datetime as datetime
import types
import scipy.io
import matplotlib.gridspec as gridspec

from scipy import stats
from xformpy import xflib
import logging
import math
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import nquad

from GLD_file_tools import GLD_file_tools

import aacgmv2

def data_grid_at(in_time, gld=None):
    # P_A = 5e3
    # P_B = 1e5
    # tpeak  = np.log(P_A/P_B)/(P_A - P_B)
    # Ipeak = np.exp(-P_A*tpeak) - np.exp(-P_B*tpeak)
    # Ipeak2Io = 1.0/Ipeak
    Ipeak2Io = 1.2324    # Conversion between peak current and Io
                         # (i.e., normalize the exponential terms)

#     print np.shape(times_to_do)
    print(f"loading flashes at {in_time}")
    data_grid = []
#     for in_time in times_to_do:

    Kpm = Kpmax[Kpmtimes == in_time]
    Kp_cur = Kp[np.where(Ktimes == in_time)[0]]

    flashes, flash_times = gld.load_flashes(in_time, lookback_time)
    print(np.shape(flashes))
    if flashes is not None:
        if CGM:
            pre_cgm_happy_inds = (np.abs(flashes[:,7]) < 90.) & (np.abs(flashes[:,8]) < 360.) 
            I_pre = np.array(flashes[pre_cgm_happy_inds,9])
            ft_pre = flash_times[pre_cgm_happy_inds]

            # cgmlat, cgmlon = aacgmv2.convert(flashes[pre_cgm_happy_inds,7], flashes[pre_cgm_happy_inds,8], 5.0*np.ones_like(flashes[pre_cgm_happy_inds,7]))
            geolats = (flashes[pre_cgm_happy_inds, 7])
            geolons = (flashes[pre_cgm_happy_inds, 8])
            geoalts = (5.0*np.ones_like(geolats)).tolist()
            
            cgmlat = np.zeros([len(geolats),1])
            cgmlon = np.zeros([len(geolats),1])
            mlts   = np.zeros([len(geolats),1])

            # Do the loop here, since the vectorized version is having trouble
            for x in range(len(geolats)):
                cgmlat[x], cgmlon[x], mlts[x] = aacgmv2.convert_latlon(geolats[x], geolons[x], 5.0, ft_pre[x], 'G2A')

            happy_inds = ~np.isnan(cgmlat).flatten()
            cgmlat = cgmlat[happy_inds]
            cgmlon = cgmlon[happy_inds]
            mlts   = mlts[happy_inds]
            cgmlon[cgmlon < 0] += 360.

            I = np.atleast_2d(I_pre[happy_inds]*Ipeak2Io).T
            print(np.shape(cgmlat), np.shape(cgmlon), np.shape(mlts), np.shape(I))
            data_grid = np.hstack([cgmlat, cgmlon, mlts, I, Kpm*np.ones_like(I), Kp_cur*np.ones_like(I)])

            return data_grid

        else:
            for flash, flashtime in zip(flashes, flash_times):

                glat = flash[7]
                glon = flash[8]
                # I    = flash[9]*Ipeak2Io  # Added after stats_v6
                I    = flash[9]*Ipeak2Io  # Added after stats_v6
                # Get location in geomagnetic coordinates
                mloc = xf.rllgeo2rllmag([1.0, glat, glon], flashtime)

                # Get MLT:
                mlt = xf.lon2MLT(flashtime, mloc[2])

                data_grid.append([mloc[1], mloc[2], mlt, I, Kpm, Kp_cur])
            data_grid = np.array(data_grid)

            return data_grid
    else:
        return None

if __name__ == "__main__":

    print("hey dude, hey")
    xf = xflib.xflib(lib_path='xformpy/libxformd.so')

    # lookback_time = datetime.timedelta(hours=3)
    lookback_time = datetime.timedelta(hours=3)

    # The range of times we'll do:
    start_day = datetime.datetime(2013,1,1,0,0,0)
    stop_day = datetime.datetime(2013,1,1,3,0,0)

    CGM = True # Convert from geo to CGM?

    GLD_path = "/Volumes/lairdata/lightningdata/From Alexandria/GLD_cleaned/ASCII";
    gld = GLD_file_tools(GLD_path, prefix='GLD')
    gld.refresh_directory()

    # Get Kp data
    Ktimes, Kp = load_Kp('data/indices/Kp_1999_2018.dat')
    Ktimes = [k + datetime.timedelta(minutes=90) for k in Ktimes]  # 3-hour bins; the original script labeled them in the middle of the bin
    Ktimes = np.array(Ktimes)
    Kp = np.array(Kp)

    # Get Kpmax -- max value of Kp over the last 24 hours (8 bins):
    Kpmax = np.max([Kp[0:-8],Kp[1:-7],Kp[2:-6], Kp[3:-5], Kp[4:-4],Kp[5:-3],Kp[6:-2], Kp[7:-1], Kp[8:]],axis=0)
    Kpmtimes = Ktimes[8:]


    tasklist = Ktimes[(Ktimes > start_day) & (Ktimes <= stop_day)]

    for intime in tasklist:
        print(intime)

        fig, ax = plt.subplots(1,1)
        CGM = True
        datagrid = data_grid_at(intime, gld)
        print(np.shape(datagrid))
        ax.plot(datagrid[:,1], datagrid[:,0],'.')        
        
        CGM = False
        datagrid = data_grid_at(intime, gld)
        print(np.shape(datagrid))
        ax.plot(datagrid[:,1], datagrid[:,0],'.')        

        plt.show()

        # Friday night: data_grid_at seems happy! Data looks as expected for dipole and CGM. Happy happy.
        # next step: analyze_flashes. Can you do this on your local machine, or do you need to
        # prep the inputs and push them over to nansen? Ideally you can reduce the GLD data
