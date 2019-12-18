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
from scipy.interpolate import interp1d
from GLD_file_tools import GLD_file_tools

import aacgmv2

from solar import daynight_terminator



def data_grid_at(in_time, lookback_time, gld=None):
    # P_A = 5e3
    # P_B = 1e5
    # tpeak  = np.log(P_A/P_B)/(P_A - P_B)
    # Ipeak = np.exp(-P_A*tpeak) - np.exp(-P_B*tpeak)
    # Ipeak2Io = 1.0/Ipeak
    Ipeak2Io = 1.2324    # Conversion between peak current and Io
                         # (i.e., normalize the exponential terms)
    # Ipeak2Io = 1
#     print np.shape(times_to_do)
    print(f"loading flashes at {in_time}")
    data_grid = []

    Ktimes, Kp = load_Kp('data/indices/Kp_1999_2018.dat')
    Ktimes = [k + datetime.timedelta(minutes=90) for k in Ktimes]  # 3-hour bins; the original script labeled them in the middle of the bin
    Ktimes = np.array(Ktimes)
    Kp = np.array(Kp)

    # Get Kpmax -- max value of Kp over the last 24 hours (8 bins):
    Kpmax = np.max([Kp[0:-8],Kp[1:-7],Kp[2:-6], Kp[3:-5], Kp[4:-4],Kp[5:-3],Kp[6:-2], Kp[7:-1], Kp[8:]],axis=0)
    Kpmtimes = Ktimes[8:]

    itrpl = interp1d([x.timestamp() for x in Kpmtimes], Kpmax, kind='nearest')
    Kpm = itrpl(in_time.timestamp())
    itrpl = interp1d([x.timestamp() for x in Ktimes], Kp, kind='nearest')
    Kp_cur = itrpl(in_time.timestamp())

    flashes, flash_times = gld.load_flashes(in_time, lookback_time)




    print(f'Loaded {np.shape(flashes)[0]} flashes')
    if flashes is not None:

        # ------------------- day / night calculation --------------------       
        # Threshold day / night via terminator, in geographic coordinates:
        is_day = np.zeros_like(flash_times, dtype='bool')
        t0 = in_time - lookback_time
        t1 = in_time
        dt_daynite = datetime.timedelta(minutes=10) # Timestep to generate day/nite terminator
        tbreaks = [datetime.datetime.fromtimestamp(x) for x in 
                            np.arange(t0.timestamp(), t1.timestamp(), dt_daynite.seconds)]


        for ta in tbreaks:
            tcenter = ta + dt_daynite/2
            tb = ta + dt_daynite
            # print(f'{ta} and {tb}')
            # print(f'flash times contain dates between {min(flash_times)} and {max(flash_times)}')
            # print( np.where((flash_times >=ta) & (flash_times < tb))[0])
            hits = np.where((flash_times >=ta) & (flash_times < tb))[0]
            if len(hits) > 0:
        
                local_times = flash_times[hits]
                local_lons = flashes[hits,8]
                local_lats = flashes[hits,7]

                # Get terminator, in geographic coordinates
                tlons, tlats, tau, dec = daynight_terminator(tcenter, 1, -180, 180)

                # Lon to lat interpolator
                interpy = interp1d(tlons, tlats,'linear', fill_value='extrapolate')

                thresh_lats = interpy(local_lons)
                # fig, ax = plt.subplots(1,1)
                # ax.plot(tlons, tlats)
                # plt.show()

                
                if dec > 0: 
                    is_day[hits] = local_lats > thresh_lats
                else:
                    is_day[hits] = local_lats < thresh_lats


        print(f'{np.sum(is_day)} day bins, {np.sum(~is_day)} night bins')
        if CGM:
        # ----------------------- CGM coordinates -----------------------
            # pre_cgm_happy_inds = (np.abs(flashes[:,7]) < 90.) & (np.abs(flashes[:,8]) < 360.) 
            I_pre = np.array(flashes[:,9])
            ft_pre = flash_times[:]

            # cgmlat, cgmlon = aacgmv2.convert(flashes[pre_cgm_happy_inds,7], flashes[pre_cgm_happy_inds,8], 5.0*np.ones_like(flashes[pre_cgm_happy_inds,7]))
            geolats = (flashes[:, 7])
            geolons = (flashes[:, 8])
            geoalts = (5.0*np.ones_like(geolats)).tolist()
            
            cgmlat = np.zeros([len(geolats)])
            cgmlon = np.zeros([len(geolats)])
            mlts   = np.zeros([len(geolats)])

            # Do the loop here, since the vectorized version is having trouble
            for x in range(len(geolats)):
                cgmlat[x], cgmlon[x], _ = aacgmv2.convert_latlon(geolats[x], geolons[x], 5.0, ft_pre[x], 'G2A')
                
                # Theirs is probably more correct than mine! But it's slow...
                mlts[x] = aacgmv2.convert_mlt(cgmlon[x], ft_pre[x], False)

            happy_inds = ~np.isnan(cgmlat).flatten()

            cgmlat = cgmlat[happy_inds]
            cgmlon = cgmlon[happy_inds]
            mlts   = mlts[happy_inds]
            is_day = is_day[happy_inds]
            cgmlon[cgmlon < 0] += 360.

            I = I_pre[happy_inds]*Ipeak2Io
            # print(Kpm)
            # print(np.shape(cgmlat), np.shape(cgmlon), np.shape(mlts), np.shape(I), Kpm, Kp_cur)
            # print(mlts)
            # print(np.shape(cgmlat), np.shape(cgmlon), np.shape(mlts), np.shape(I), np.shape(Kpm*np.ones_like(I)))
            data_grid = np.vstack([cgmlat, cgmlon, mlts, I, Kpm*np.ones_like(I), Kp_cur*np.ones_like(I), is_day]).T
            # print(data_grid[:,2])
            return data_grid

        else:
        # ----------------------- Magnetic Dipole coordinates -----------------------
            for flash, flashtime, day_flag in zip(flashes, flash_times, is_day):

                glat = flash[7]
                glon = flash[8]
                # I    = flash[9]*Ipeak2Io  # Added after stats_v6
                I    = flash[9]*Ipeak2Io  # Added after stats_v6
                # Get location in geomagnetic coordinates
                mloc = xf.rllgeo2rllmag([1.0, glat, glon], flashtime)

                # Get MLT:
                mlt = xf.lon2MLT(flashtime, mloc[2])

                data_grid.append([mloc[1], mloc[2], mlt, I, Kpm, Kp_cur, day_flag])

            data_grid = np.array(data_grid)
            # print(data_grid[:,2])
            return data_grid
    else:
        return None



# Output power space:
def analyze_flashes(data_grid, in_time):
    Ipeak2Io = 1.2324    # Conversion between peak current and Io

    outdata = dict()
    dg2 = np.array(data_grid)

    # Quantize data into lat, lon, and MLT bins:
    dg2[:,0] = np.digitize(dg2[:,0], gridlats)
    dg2[:,1] = np.digitize(np.mod(dg2[:,1], 360.0) - 180.0, gridlons)
    dg2[:,2] = (dg2[:,2] > 6) & (dg2[:,2] <= 18)   # Is day?

    # hist_bins = np.arange(0,24.5, 0.5)
    flash_map = np.zeros([len(gridlats), len(gridlons)])
    cur_map   = np.zeros([len(gridlats), len(gridlons)])
    pwr_map   = np.zeros([len(gridlats), len(gridlons)])
    mlt_hist, _  = np.histogram(data_grid[:,2], hist_bins)

    # Io_bins = np.linspace(0,1000,101)
    Io_bins = np.unique(np.round(pow(10,np.linspace(0,3,144))))*Ipeak2Io # 101 log-spaced dividers
    Io_hist, _ = np.histogram(np.abs(data_grid[:,3]), Io_bins)

    # Bin total current by lat and lon
    day_bins = np.zeros([len(gridlats), len(gridlons)])
    nite_bins= np.zeros([len(gridlats), len(gridlons)])

    
    for row in dg2:
        # Power stencils are proportional to current squared;
        # GLD currents are in kA
        # if row[2]:
        if row[-1]: # is_day flag
            day_bins[int(row[0]), np.mod(int(row[1]), 360)] += pow(row[3]*1e3, 2.0)
        else:
            # print(row[0], row[1], np.mod(row[1], 360))
            nite_bins[int(row[0]), np.mod(int(row[1]), 360)] += pow(row[3]*1e3, 2.0)
        
        # Flash count histogram
        flash_map[int(row[0]), np.mod(int(row[1]), 360)] += 1
        
        # 2d current histogram
        cur_map[int(row[0]), np.mod(int(row[1]), 360)] += pow(row[3]*1e3, 2.0)



    # --------- Power stencil map
    day_todo = np.where(day_bins > 0)
    nite_todo = np.where(nite_bins > 0)

    for isday in [False, True]:
        if isday:
            todo = np.where(day_bins > 0)
        else:
            todo = np.where(nite_bins > 0)
        for latind, lonind in zip(todo[0], todo[1]):
            if (np.abs(gridlats[latind]) >= stencil_lats[0]) & (np.abs(gridlats[latind]) <= stencil_lats[-1]):

                if isday:
                    key = (np.round(gridlats[latind]), 12)
                else:
                    key = (np.round(gridlats[latind]), 0)
                if key in inp_pwr_dict:
                    stencil = inp_pwr_dict[key]
                    if isday:
                        pwr = day_bins[latind, lonind]
                    else:
                        pwr = nite_bins[latind, lonind]
                    latleft = int(latind - cell_lat_offset)
                    latright = int(latind + cell_lat_offset-1)
                    lonleft = int(lonind - cell_lon_offset)
                    lonright =int(lonind + cell_lon_offset-1)

                    if lonleft < 0:
                        # Wrap around left:
                        pwr_map[latleft:latright, 0:lonright] += \
                                stencil[:, np.abs(lonleft):]*pwr
                        pwr_map[latleft:latright, (len(gridlons) - np.abs(lonleft)):] += \
                                stencil[:,0:np.abs(lonleft)]*pwr
                    elif lonright >= len(gridlons):
                        # wrap around right:
                        pwr_map[latleft:latright, lonleft:len(gridlons)] += \
                            stencil[:,0:len(gridlons) - lonleft]*pwr
                        pwr_map[latleft:latright, 0:np.abs(lonright) - len(gridlons)] += \
                            stencil[:,len(gridlons) - lonleft:]*pwr
                    else:
                        pwr_map[latleft:latright, lonleft:lonright] += stencil*pwr
    # -----




    # Roll the output data arrays to get [-180, 180] instead of [0, 360]
    outdata['pwr_map']   = np.roll(pwr_map,  int(len(gridlons)/2), axis=1)
    outdata['flash_map'] = np.roll(flash_map,int(len(gridlons)/2), axis=1)
    outdata['cur_map']   = np.roll(cur_map , int(len(gridlons)/2), axis=1)
    outdata['mlt_hist']  = mlt_hist
    outdata['Io_hist']   = Io_hist
    outdata['in_time']   = in_time
    return outdata


# Convert the Matlab coastline datafile to geomagnetic coordinates:
def get_coast_mag(itime):
    # xf = xflib.xflib(lib_path='/shared/users/asousa/WIPP/3dWIPP/python/libxformd.so')
    coastlines = scipy.io.loadmat('data/coastlines.mat')

    coast_lat_mag = np.zeros(len(coastlines['lat']))
    coast_lon_mag = np.zeros(len(coastlines['long']))

    for ind, (lat, lon) in enumerate(zip(coastlines['lat'], coastlines['long'])):
        if np.isnan(lat) or np.isnan(lon):
            coast_lat_mag[ind] = np.nan
            coast_lon_mag[ind] = np.nan
        else:
            tmpcoords = [1, lat[0], lon[0]]
            tmp_mag = xf.rllgeo2rllmag(tmpcoords, itime)
            coast_lat_mag[ind] = tmp_mag[1]
            coast_lon_mag[ind] = tmp_mag[2]

    # Loop around for -180 + 180 ranges
    coast_lat_mag = np.concatenate([coast_lat_mag, coast_lat_mag[coast_lon_mag > 180.0]])
    coast_lon_mag = np.concatenate([coast_lon_mag, (coast_lon_mag[coast_lon_mag > 180.0] - 360)])

    # Toss in some NaNs to break up the continents
    for ind in range(len(coast_lat_mag) -1):
        if ((np.abs(coast_lat_mag[ind+1] - coast_lat_mag[ind]) > 5) or
           (np.abs(coast_lon_mag[ind+1] - coast_lon_mag[ind]) > 5)):
            coast_lat_mag[ind] = np.nan
            coast_lon_mag[ind] = np.nan

    return coast_lat_mag, coast_lon_mag



def plot_pwr_data(outdata):
    # --------------- Latex Plot Beautification --------------------------
    fig_width = 12 
    fig_height = 6
    fig_size =  [fig_width+1,fig_height+1]
    params = {'backend': 'ps',
              'axes.labelsize': 14,
              'font.size': 14,
              'legend.fontsize': 14,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': False,
              'figure.figsize': fig_size}
    plt.rcParams.update(params)
    # --------------- Latex Plot Beautification --------------------------
    
    pwr_map = outdata['pwr_map']
    flash_map = outdata['flash_map']
    cur_map = outdata['cur_map']
    mlt_hist = outdata['mlt_hist']
    in_time = outdata['in_time']

    clims=[0, 4] # log space

    # Plot flash location with integrated current vs lat and lon:
    fig = plt.figure()
    gs = gridspec.GridSpec(2,3,width_ratios=[0.5,10,0.25],
                       height_ratios=[10,1])
    # gs.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs[0:-1,1:-1])  # main figure
    ax0 = plt.subplot(gs[0:-1,0])   #
    ax2 = plt.subplot(gs[-1,1:-1])
    cbar_ax = plt.subplot(gs[:, -1])
    pwr_bylat = np.sum(pwr_map, axis=1)/lookback_time.total_seconds()
    pwr_bylon = np.sum(pwr_map, axis=0)/lookback_time.total_seconds()
    
    logpwr = np.log10(pwr_map/3600./3.0)  # Energy per second, in log space.
    logpwr[np.isinf(logpwr)] = -10


    p = ax1.pcolorfast(gridlons, gridlats, logpwr, cmap = plt.get_cmap('viridis'))
    p.set_clim(clims)

    cb = plt.colorbar(p, cax=cbar_ax)

    cb.set_label('Average Energy Flux [J/sec]')
    cticks = np.arange(clims[0],clims[1] + 1)
    cb.set_ticks(cticks)
    cticklabels = ['$10^{%d}$'%k for k in cticks]
    cb.set_ticklabels(cticklabels)



    coast_lat_mag, coast_lon_mag = get_coast_mag(in_time)
    ax1.plot(coast_lon_mag, coast_lat_mag, 'w')
    ax1.set_xlim([-180, 180])
    ax1.set_ylim([-90, 90])
    p.set_clim([0,4])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax0.plot(pwr_bylat,gridlats)
    ax0.set_ylim([-90,90])
    ax0.set_xticks([])
    ax0.set_ylabel('Latitude (magnetic)')
    ax2.plot(gridlons, pwr_bylon)
    ax2.set_xlim([-180,180])
    ax2.set_yticks([])
    ax2.set_xlabel('Longitude (magnetic)')

    ax1.set_title( in_time.isoformat() )
    fig.tight_layout()

    return fig, ax0, ax1, ax2





# def do_thing():
if __name__ == "__main__":
    print("hey dude, hey")
    xf = xflib.xflib(lib_path='xformpy/libxformd.so')

    lookback_time = datetime.timedelta(hours=3)
    # lookback_time = datetime.timedelta(minutes=5)

    # The range of times we'll do:
    start_day = datetime.datetime(2014,12,14,0,0,0)
    stop_day = datetime.datetime(2014,12,14,21,0,0)
    

    CGM = True # True for geo -> CGM; false for geo -> MAG
                # (CGM = the equivalent magnetic dipole coordinate, given the IGRF-deformed field)

    dump_path = 'outputs/GLDstats_v10_CGM/data'
    fig_path = 'outputs/GLDstats_v10_CGM/figures'

    # GLD_path = "/Volumes/lairdata/lightningdata/From Alexandria/GLD_cleaned/ASCII";
    GLD_path = "data/"
    gld = GLD_file_tools(GLD_path, prefix='FAKEGLD')
    gld.refresh_directory()
    # Get Kp data
    # Ktimes, Kp = load_Kp('data/indices/Kp_1999_2018.dat')
    # Ktimes = [k + datetime.timedelta(minutes=90) for k in Ktimes]  # 3-hour bins; the original script labeled them in the middle of the bin
    # Ktimes = np.array(Ktimes)
    # Kp = np.array(Kp)

    # # Get Kpmax -- max value of Kp over the last 24 hours (8 bins):
    # Kpmax = np.max([Kp[0:-8],Kp[1:-7],Kp[2:-6], Kp[3:-5], Kp[4:-4],Kp[5:-3],Kp[6:-2], Kp[7:-1], Kp[8:]],axis=0)
    # Kpmtimes = Ktimes[8:]


# ---------------------- Power database setup --------------------------
    pwr_db_path = 'data/pwr_db_20deg_spread.pklz';

    # Load input power database:
    with gzip.open(pwr_db_path,'rb') as f:
        inp_pwr_dict = pickle.load(f, encoding='latin1')

    tmp = np.array(sorted([k for k in inp_pwr_dict.keys() if not isinstance(k, str)]))

    # Flip the stencils for southern hemisphere
    for k in tmp:
        stencil = inp_pwr_dict[tuple(k)]
        newkey = (-1*k[0], k[1])
        inp_pwr_dict[newkey] = np.flipud(stencil)

    tmp = np.array(sorted([k for k in inp_pwr_dict.keys() if not isinstance(k, str)]))
    stencil_lats = np.unique(tmp[:,0]); stencil_MLTs = np.unique(tmp[:,1])

    cellsize = inp_pwr_dict['cellsize']
    cell_lat_offset = np.round(inp_pwr_dict['lat_spread']/inp_pwr_dict['cellsize'])
    cell_lon_offset = np.round(inp_pwr_dict['lon_spread']/inp_pwr_dict['cellsize'])

    gridlats = np.arange(-90, 91, cellsize)
    gridlons = np.arange(-180, 181, cellsize)

    hist_bins = np.arange(0,24.5, 0.5)

    print(f'Cell size is {cellsize} degree in lat / lon')


# ---------------------- MAIN LOOP --------------------------
    # tasklist = Ktimes[(Ktimes > start_day) & (Ktimes <= stop_day)]


    # # Debuggin'
    start_day = datetime.datetime(2015,1,2,0,0,0)
    lookback_time = datetime.timedelta(hours=24)
    # tasklist = [(start_day + datetime.timedelta(hours=int(x)))for x in np.arange(0,1,1)]
    tasklist = [start_day]
    for intime in tasklist:
        print(intime)

        datagrid = data_grid_at(intime, lookback_time, gld)

        if datagrid is not None:
            datum = analyze_flashes(datagrid, intime)
            filename = os.path.join(dump_path, intime.strftime('%m_%d_%Y_%H_%M') + '.pklz')
            print(f' datagrid has shape {np.shape(datagrid)}')
            print(f' Output filename is {filename}')

            if datum is not None:
                fig, ax0, ax1, ax2 = plot_pwr_data(datum)
                figname = os.path.join(fig_path, datum['in_time'].strftime('%m_%d_%Y_%H_%M') +'.png')
                fig.savefig(figname, ldpi=300)

                plt.close(fig)
                print(f"sum of cur_map: {np.sum(np.sum(datum['cur_map']))}")
                filename = os.path.join(dump_path, datum['in_time'].strftime('%m_%d_%Y_%H_%M') + '.pklz')
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(datum,f)



                # fig, ax = plt.subplots(1,1)
                # ax.plot(hist_bins[:-1], datum['mlt_hist'])
                # plt.show()
    # datagrid 

###
    # intime = datetime.datetime(2015,1,2,0,0,0)
    # lookback_time = datetime.timedelta(hours=24)

    # fig, ax = plt.subplots(1,1)
    # CGM = True
    # datagrid = data_grid_at(intime, gld)
    # print(np.shape(datagrid))
    # ax.plot(datagrid[:,1], datagrid[:,0],'.')        
    # print(f'data grid (CGM) is {np.shape(datagrid)}')
    # print(f"I_sum CGM is {np.sum(np.abs(datagrid[:,3]))}")
    # # plt.show()
    # CGM = False
    # datagrid = data_grid_at(intime, gld)
    # print(np.shape(datagrid))
    # ax.plot(datagrid[:,1], datagrid[:,0],'.')     
    # print(f'data grid (MAG) is {np.shape(datagrid)}')   
    # print(f"I_sum dipole is {np.sum(np.abs(datagrid[:,3]))}")
    # plt.show()

        # Friday night: data_grid_at seems happy! Data looks as expected for dipole and CGM. Happy happy.
        # next step: analyze_flashes. Can you do this on your local machine, or do you need to
        # prep the inputs and push them over to nansen? Ideally you can reduce the GLD data


        # Monday night (12/16): Confirmed MLT is working correctly; looks like data grid and analysis are
        # pretty happy. Run 'em now? Next, look up what you did for the actual magnetosphere stencils --
        # the power stencils here are just the top-of-the-ionosphere models

