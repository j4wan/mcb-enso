### PURPOSE: Script to compute SPMM and NPMM
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/05/2026
##################################################################################################################
#%% IMPORT LIBRARIES, DATA, AND FORMAT
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import glob
from importlib import reload #to use type reload(fun)
import matplotlib.patches as mpatches
from scipy import signal
from scipy import stats
import function_dependencies as fun
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import datetime
import os
import dask
from eofs.xarray import Eof
import sys
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import statsmodels.api as sm

plt.ion();


dask.config.set({"array.slicing.split_large_chunks": False})

#run this line in console before starting ipython if getting OMP_NUM_THREADS error
os.environ["OMP_NUM_THREADS"] = "1"

##################################################################################################################
## THIS SCRIPT READS IN ONE ENSEMBLE OF EXPERIMENTS AT A TIME. ##
## WHICH EXPERIMENT ARE YOU READING IN? ##
month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
year_init = input('Which initialization year are you reading in (1997, 2015, 2019?): ')
sensitivity_opt = input('Sensitivity run (y or n)?: ') # y for 05-1997 and 05-2015 only, else n
mcb_keys = ['06-02','06-08','06-11','09-02','09-11','12-02']
## UNCOMMENT THESE OPTIONS FOR DEMO ##
month_init = '05'
year_init = '2015'
sensitivity_opt = 'y'
mcb_keys = ['06-02']
##################################################################################################################

## DEFINE FUNCTIONS FOR ANALYSIS ##
# 1) djf_mean_annual: calculate DJF means for each year (D is defined in year t-1 and JF in year t)
def djf_mean_annual(data):
    # data: monthly dataarray over which you want to calculate DJF means
    ## Adapted from: https://stackoverflow.com/questions/64976340/keeping-time-series-while-grouping-by-season-in-xarray
    ## Resample monthly data into seasonal means (DJF has months=12, MAM has months=3, JJA has months=6, and SON has months=9)
    seasonal_xr = data.resample(time='QS-DEC').mean(dim='time')
    # Select only DJF (one season per year)
    djf_xr = seasonal_xr.loc[{'time':[t for t in pd.to_datetime(seasonal_xr.time.values) if (t.month==12)]}]
    # Reassign time such that D is defined in year t-1 and JF in year t
    djf_xr = djf_xr.assign_coords(time=pd.to_datetime(djf_xr.time).year+1)
    return djf_xr


## 2) xarray_linear_detrend: detrend xarray along time dimension
# # Adapted from Callahan & Mankin (2023) Observed_ENSO_Indices.ipynb and CMIP6_ENSO_Indices
def xarray_linear_detrend(data):
    # detrends a three-dimensional
    # (time,lat,lon)
    # xarray dataarray separately at 
    # each grid point
    # easy to do, but slow, with a loop
    # so this is a vectorized
    # way of doing it 
    # https://stackoverflow.com/questions/38960903/applying-numpy-polyfit-to-xarray-dataset

    def linear_trend(x, y):
        pf = np.polyfit(x, y, 1)
        return xr.DataArray(pf[0])
    def intercepts(x, y):
        pf = np.polyfit(x, y, 1)
        return xr.DataArray(pf[1])
    
    tm = data.time
    lt = data.lat
    ln = data.lon
    timevals = xr.DataArray(np.arange(1,len(tm)+1,1),
                        coords=[tm],
                        dims=["time"])
    # timevals = data['time.year']+(data.time.dt.dayofyear/365)
    # timevals = timevals.expand_dims(lat=lt,lon=ln)
    # timevals = timevals.transpose("time","lat","lon")
    
    trends = xr.apply_ufunc(linear_trend,
                            timevals,data,
                            vectorize=True,
                            input_core_dims=[["time"],["time"]])
    intcpts = xr.apply_ufunc(intercepts,
                             timevals,data,
                             vectorize=True,
                             input_core_dims=[["time"],["time"]])
    
    predicted_vals = (intcpts + trends*timevals).transpose("time","lat","lon")
    detrended_data = data - predicted_vals
    return detrended_data, predicted_vals

## 3) eof_calc: Calculate EOF and PC
def eof_calc(data, nmodes):
    # data: xarray with dimensions [time, lat, lon] (time must be first dimension)
    # nmodes: number of modes you want to calculate
    # returns EOFs, variance fraction, and PCs
    eof_solver = Eof(data)
    eofs = eof_solver.eofs(neofs=nmodes)
    vf = eof_solver.varianceFraction(neigs=nmodes)
    pcs = eof_solver.pcs(pcscaling=1,npcs=nmodes)
    
    return eofs, vf, pcs

## 4) remove_cov: Compute EOF, perform linear regression, remove ENSO signal
def remove_cov(data, PC1):
    # data: xarray with dimensions [time, lat, lon] (time must be first dimension)
    # PC1: principal component of EOF you wish to remove from the data [time,]
    # Extract values from data
    Y = data.values
    # Reshape
    nt, nlat, nlon = data.shape
    Y_2d = Y.reshape(nt, -1)
    # Regression coefficient
    a = np.dot(PC1, Y_2d) / np.dot(PC1, PC1)
    # Reconstruct ENSO signal and remove from original data
    enso_part = np.outer(PC1, a)
    Y_clean = Y_2d - enso_part
    # Reshape to original data
    Y_clean = Y_clean.reshape(nt, nlat, nlon)
    Y_resid = enso_part.reshape(nt, nlat, nlon)
    # Convert to xarray
    data_processed = xr.DataArray(data=Y_clean, coords={'time':data.time,'lat':data.lat,'lon':data.lon},dims=['time','lat','lon'])
    data_resid = xr.DataArray(data=Y_resid, coords={'time':data.time,'lat':data.lat,'lon':data.lon},dims=['time','lat','lon'])

    return data_processed, data_resid


## 5) pc_scaling: Scale PC1 with Nino3:
def pc_scaling(data, scaler, eof, pc, nmodes):
    # Scale PC1 sign to be positively correlated with Nino3
    for i in np.arange(0,nmodes,1):
        # Want to compute correlation with Nino3 without MCB simulation
        corrcoef = np.corrcoef(pc.isel(time=slice(None,len(data.time)))[:,i].values,scaler.isel(time=slice(None,len(data.time))).values)
        # we want mode 1 to be positively correlated with nino3
        if ((i == 0) & (corrcoef[0][1]<0)):
            scaling = -1
        elif ((i == 0) & (corrcoef[0][1]>=0)):
            scaling = 1
        # and mode 2 to be negatively corelated with nino3
        elif ((i == 1) & (corrcoef[0][1]<0)):
            scaling = -1
        elif ((i == 1) & (corrcoef[0][1]>=0)):
            scaling = -1
        else:
            print("ERROR")
            sys.exit()
        # print(corrcoef[0][1])
        eof[i,:,:] = eof[i,:,:].values*scaling
        pc[:,i] = pc[:,i].values*scaling
        #print(scaling)
    return eof, pc

##################################################################################################################

## READ IN DATA
# Get list of control ensemble members
if year_init=='1997':
    yr_init = ['1996','1997']
elif year_init=='2015':
    yr_init = ['2014','2015']
elif year_init=='2019':
    yr_init = ['2019','2020']
ctrl_files = []
for yr in yr_init:
    ctrl_files = ctrl_files + glob.glob('/_data/realtime/b.e21.BSMYLE.f09_g17.'+yr+'*-'+month_init+'.*')
ctrl_members = []
for i in ctrl_files:
    start = i.find('f09_g17.') + len('f09_g17.')
    tmp = i[start:None]
    if tmp not in ctrl_members:
        ctrl_members.append(tmp)
ctrl_members = sorted(ctrl_members)
print(ctrl_members) 

# Get list of MCB ensemble members
mcb_sims = {}
if sensitivity_opt=='y':
    mcb_keys = ['06-02','06-08','06-11','09-02','09-11','12-02']
    for key in mcb_keys:
        for yr in yr_init:
            mcb_files = []
            mcb_files = mcb_files + glob.glob('/_data/MCB/b.e21.BSMYLE.f09_g17.MCB*'+yr+'*-'+month_init+'_'+key+'.*')
        mcb_members = []
        for i in mcb_files:
            start = i.find('f09_g17.MCB') + len('f09_g17.MCB.')
            tmp = i[start:None]
            if tmp not in mcb_members:
                mcb_members.append(tmp)
        mcb_members = sorted(mcb_members)
        print(mcb_members)
        mcb_sims[key] = mcb_members     
elif sensitivity_opt=='n':
    mcb_keys=['']
    for key in mcb_keys:
        mcb_files = []
        for yr in yr_init:    
            mcb_files = mcb_files + glob.glob('/_data/MCB/b.e21.BSMYLE.f09_g17.MCB.'+yr+'*-'+month_init+'.*')
        mcb_members = []
        for i in mcb_files:
            start = i.find('f09_g17.MCB') + len('f09_g17.MCB.')
            tmp = i[start:None]
            if tmp not in mcb_members:
                mcb_members.append(tmp)
        mcb_members = sorted(mcb_members)
        print(mcb_members)
        mcb_sims[key] = mcb_members 


# # Get interesction of control and MCB ensemble members so we only keep members that are in both
intersect_members = ctrl_members[0:len(mcb_members)]

# Create variable subset list
atm_varnames_monthly_subset = ['TS']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day


## READ IN CONTROL SMYLE-FOSI HISTORICAL SIMULATIONS
# SMYLE input data can be found at the SMYLE archive on NSF-NCAR HPC Derecho
# /glade/campaign/cesm/development/espwg/SMYLE/archive/
data_dir='/_data/SMYLE-FOSI/regrid/' # This is just a placeholder
# Read in potential temperature and select surface layer only to reduce file sice
ocn_temp_hist_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(data_dir+'r288x192.g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005.pop.h.TEMP.030601-036812.nc'))).TEMP.isel(z_t=0)
# Reassign time values to be between 1958-2020 (Yeager et al., 2022)
start_yr=1958
delta_yr = start_yr - int(ocn_temp_hist_xr.time[0].dt.year.values)
ocn_temp_hist_xr = ocn_temp_hist_xr.assign_coords(time=ocn_temp_hist_xr['time']+datetime.timedelta(days=365*delta_yr))
ocn_temp_hist_xr = ocn_temp_hist_xr.assign_coords(time=ocn_temp_hist_xr.indexes['time'].to_datetimeindex())
# Fix any rounding errors for lat,lon grid so they match SMYLE exactly
sample_path = '/_data/realtime/b.e21.BSMYLE.f09_g17.2015-05.001/atm/proc/tseries/month_1/b.e21.BSMYLE.f09_g17.2015-05.001.cam.h0.TS.201505-201704.nc'
smyle_lat = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(sample_path))).lat
smyle_lon = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(sample_path))).lon
ocn_temp_hist_xr = ocn_temp_hist_xr.assign_coords(lat= smyle_lat, lon= smyle_lon)


## COMPUTE LONG TERM STANDARD DEVIATION AND MONTHLY CLIMATOLOGY MEAN FROM 1970-2017
# Subset time from 1970-2017
hist_window = ocn_temp_hist_xr.loc[{'time':[t for t in pd.to_datetime(ocn_temp_hist_xr.time.values) if (t.year>=1970)&\
                                                         (t.year<=2017)]}]

# Create formatted historical time series to append to control and MCB runs
hist_ext = hist_window.isel(time=hist_window['time.year']<=2015)

# Calculate monthly climatological mean
hist_clim_ens_mean = hist_ext.groupby('time.month').mean()
hist_clim_ens_sd = hist_ext.std(dim='time')


## READ IN CONTROL SIMULATION & PRE-PROCESS
# ATM
atm_monthly_ctrl={}
ts_ctrl_anom={}
ts_ctrl_ext_anom={} #extended time series appending historical SMYLE
prect_ctrl_anom = {}
prect_ctrl_ext_anom={} #extended time series appending historical SMYLE

ctrl_keys=['']
for key in ctrl_keys:
    atm_monthly_ctrl_single_mem = {}
    for m in intersect_members:
        print(m)
        dir_ctrl = '/_data/realtime/b.e21.BSMYLE.f09_g17.'+m+'/atm/proc/tseries/month_1'
        file_subset_ctrl = []
        for var in atm_varnames_monthly_subset:
            pattern = "."+var+"."
            var_file_ctrl = [f for f in os.listdir(dir_ctrl) if pattern in f]
            file_subset_ctrl.append(dir_ctrl+'/'+var_file_ctrl[0])
        atm_monthly_ctrl_single_mem[m] = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_mfdataset(file_subset_ctrl)))
    # Combine all files into one xarray dataset with ensemble members as a new dimension
    atm_monthly_ctrl[key] = xr.concat(list(map(atm_monthly_ctrl_single_mem.get, intersect_members)),pd.Index(intersect_members,name='member'))
    # Convert time to datetime index
    atm_monthly_ctrl[key] = atm_monthly_ctrl[key].assign_coords(time=atm_monthly_ctrl[key].indexes['time'].to_datetimeindex())
    ## TS
    # Convert from K to C
    atm_monthly_ctrl[key] = atm_monthly_ctrl[key].assign(TS=atm_monthly_ctrl[key]['TS']-273.15)
    atm_monthly_ctrl[key]['TS'].attrs['units'] = '°C'
    ##DRIFT CORRECTION
    # Compute drift correction anomaly
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_ctrl_copy = atm_monthly_ctrl[key]['TS']*1
    ## TS
    ts_ctrl_anom[key] = ts_ctrl_copy.groupby('time.month') - hist_clim_ens_mean
    # Reassign units
    ## TS
    ts_ctrl_anom[key].attrs['units']='\N{DEGREE SIGN}C'



## READ IN MCB SIMULATIONS & PRE-PROCESS
# ATM
atm_monthly_mcb={}
ts_mcb_anom={}
ts_mcb_ext_anom={} #extended time series appending historical SMYLE
prect_mcb_anom = {}
prect_mcb_ext_anom = {} #extended time series appending historical SMYLE
ts_mcb_anom_std={}
ts_mcb_anom_sem={}
for key in mcb_keys:
    atm_monthly_mcb_single_mem = {}
    for m in mcb_sims[key]:
        print(m)
        dir_mcb = glob.glob('/_data/MCB/b.e21.BSMYLE.f09_g17.MCB*'+m+'/atm/proc/tseries/month_1')[0]
        file_subset_ctrl = []
        file_subset_mcb = []
        for var in atm_varnames_monthly_subset:
            pattern = "."+var+"."
            var_file_mcb = [f for f in os.listdir(dir_mcb) if pattern in f]
            file_subset_mcb.append(dir_mcb+'/'+var_file_mcb[0])
        atm_monthly_mcb_single_mem[m] = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_mfdataset(file_subset_mcb)))
    # Combine all files into one xarray dataset with ensemble members as a new dimension
    atm_monthly_mcb[key] = xr.concat(list(map(atm_monthly_mcb_single_mem.get, mcb_sims[key])),pd.Index(intersect_members,name='member'))
    # Convert time to datetime index
    atm_monthly_mcb[key] = atm_monthly_mcb[key].assign_coords(time=atm_monthly_mcb[key].indexes['time'].to_datetimeindex())
    # Overwrite lat, lon to match control to fix rounding errors
    atm_monthly_mcb[key] = atm_monthly_mcb[key].assign_coords(lat= atm_monthly_ctrl[ctrl_keys[0]].lat, lon= atm_monthly_ctrl[ctrl_keys[0]].lon)
    # ## TS
    # Convert from K to C
    atm_monthly_mcb[key] = atm_monthly_mcb[key].assign(TS=atm_monthly_mcb[key]['TS']-273.15)
    atm_monthly_mcb[key]['TS'].attrs['units'] = '°C'
    ##DRIFT CORRECTION
    # Compute drift correction anomaly
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_mcb_copy = atm_monthly_mcb[key]['TS']*1
    ## TS
    ts_mcb_anom[key] = ts_mcb_copy.groupby('time.month') - hist_clim_ens_mean
    # Reassign units
    ## TS
    ts_mcb_anom[key].attrs['units']='\N{DEGREE SIGN}C'



# CALCULATE MONTHLY ANOMALIES FOR SMYLE-FOSI HISTORICAL (1970-2017)
ts_hist_anom = hist_window.groupby('time.month') - hist_clim_ens_mean
ts_hist_anom.attrs['units'] = '\N{DEGREE SIGN}C'


## BIAS CORRECT SMYLE MCB SIMS TO HISTORICAL
## Compute TS difference between control and MCB simulations
ts_mcb_diff = {}
ts_mcb_anom_corrected = {}
for key in mcb_keys:
    print(key)
    # Absolute difference
    #ts_mcb_diff[key] = (ts_mcb_anom[key].mean(dim='member')-ts_ctrl_anom[''].mean(dim='member'))
    ts_mcb_diff[key] = ts_mcb_anom[key]-ts_ctrl_anom['']
    ts_mcb_anom_corrected[key] = ts_hist_anom+ts_mcb_diff[key]
    ts_mcb_anom_corrected[key].attrs['units'] = '\N{DEGREE SIGN}C'



#%% CREATE INDEX MASKS
# Get overlay mask files (area is the same for all of them so can just pick one)
seeding_mask = fun.reorient_netCDF(xr.open_dataset('/_data/sesp_mask_CESM2_0.9x1.25_v3.nc'))

# Force seeding mask lat, lon to equal the output CESM2 data (rounding errors)
seeding_mask = seeding_mask.assign_coords({'lat':atm_monthly_ctrl[ctrl_keys[0]]['lat'], 'lon':atm_monthly_ctrl[ctrl_keys[0]]['lon']})
# Subset 1 month of seeded grid cells 
seeding_mask_seed = seeding_mask.mask.isel(time=9)
# Add cyclical point for ML 
seeding_mask_seed_wrap, lon_wrap = add_cyclic_point(seeding_mask_seed,coord=seeding_mask_seed.lon)

# Define Niño 3.4 region
lat_max = 5
lat_min = -5
lon_max = -120
lon_min = -170
# Generate Niño 3.4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
nino34_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
nino34_mask_wrap, lon_wrap = add_cyclic_point(nino34_mask,coord=nino34_mask.lon)


# Define Niño 3 region
lat_max = 5
lat_min = -5
lon_max = -90
lon_min = -150
# Generate Niño 3 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
nino3_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
nino3_mask_wrap, lon_wrap = add_cyclic_point(nino3_mask,coord=nino3_mask.lon)


# Add cyclical point for ML 
nino34_mask_wrap, lon_wrap = add_cyclic_point(nino34_mask,coord=nino34_mask.lon)      
## Define Niño 4 region
lat_max = 5
lat_min = -5
lon_WP_max = -150
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 160
# Generate Niño 4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
nino4_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
nino4_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

nino4_mask = nino4_WP_mask + nino4_EP_mask
# Add cyclical point for Niño 4  
nino4_mask_wrap, lon_wrap = add_cyclic_point(nino4_mask,coord=nino4_mask.lon)
     
# Define E/C Index region
lat_max = 20
lat_min = -20
lon_WP_max = -80
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 140
# Generate Niño 3.4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
ecindex_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
ecindex_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

ecindex_mask = ecindex_WP_mask + ecindex_EP_mask
# Add cyclical point for ML 
ecindex_mask_wrap, lon_wrap = add_cyclic_point(ecindex_mask,coord=ecindex_mask.lon)


# Define NPMM region
lat_max = 30
lat_min = -20
lon_NP_max = -95
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 175
# Generate Niño 3.4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
npmm_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
npmm_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

npmm_mask = npmm_WP_mask + npmm_EP_mask
# Add cyclical point for ML 
npmm_mask_wrap, lon_wrap = add_cyclic_point(npmm_mask,coord=npmm_mask.lon)


# Define SPMM region
lat_max = 0
lat_min = -40
lon_max = -90
lon_min = -140
# Generate Niño 3 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
spmm_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
spmm_mask_wrap, lon_wrap = add_cyclic_point(spmm_mask,coord=spmm_mask.lon)


# Define tropical Pacific region (to subtract from MM indices)
lat_max = 20
lat_min = -20
lon_WP_max = -90
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 140
# Generate Niño 3.4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
trop_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
trop_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

trop_mask = trop_WP_mask + trop_EP_mask
# Add cyclical point for ML 
trop_mask_wrap, lon_wrap = add_cyclic_point(trop_mask,coord=trop_mask.lon)


# Plot masked regions
plot_proj = ccrs.PlateCarree(central_longitude=180)
fig, ax = plt.subplots(figsize=(8,6), subplot_kw={'projection': plot_proj})
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=1)
ax.coastlines(color='dimgrey', linewidth=0.8, zorder=2)
# Plot each mask with a distinct color (mask value=1 highlighted, 0 transparent)
mcb_seeding_plot = xr.where(seeding_mask_seed_wrap > 0, 1, 0)
trop_plot = xr.where(trop_mask_wrap > 0, 1, 0)
spmm_plot = xr.where(spmm_mask_wrap > 0, 1, 0)
npmm_plot = xr.where(npmm_mask_wrap > 0, 1, 0)
ax.contour(lon_wrap, seeding_mask_seed.lat.values, mcb_seeding_plot, levels=[0.5],
            colors='k', linewidths=2, transform=ccrs.PlateCarree(), zorder=3)
ax.contour(lon_wrap, trop_mask.lat.values, trop_plot, levels=[0.5],
            colors=['#fc8d62'], linewidths=2, transform=ccrs.PlateCarree(), zorder=3)
ax.contour(lon_wrap, spmm_mask.lat.values, spmm_plot, levels=[0.5],
            colors=['#8da0cb'], linewidths=2, transform=ccrs.PlateCarree(), zorder=3)
ax.contour(lon_wrap, npmm_mask.lat.values, npmm_plot, levels=[0.5],
            colors=['#66c2a5'], linewidths=2, transform=ccrs.PlateCarree(), zorder=3)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
                  color='grey', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
legend_patches = [
    mpatches.Patch(edgecolor='k', facecolor='none', linewidth=2, label='MCB'),
    mpatches.Patch(edgecolor='#fc8d62', facecolor='none', linewidth=2, label='Tropical Pacific'),
    mpatches.Patch(edgecolor='#8da0cb', facecolor='none', linewidth=2, label='SPMM'),
    mpatches.Patch(edgecolor='#66c2a5', facecolor='none', linewidth=2, label='NPMM'),
]
ax.legend(handles=legend_patches, loc='lower left', frameon=True, fontsize=9)
plt.tight_layout();


### DETREND ANOMALIES
## Load in all anomaly files at once (required for detrending)
# MCB
ts_mcb_anom_load = {}
for key in mcb_keys:
    print(key)
    ts_mcb_anom_load[key] = ts_mcb_anom_corrected[key].load()
# HISTORICAL
ts_hist_anom_load = ts_hist_anom.load()

# Define NA land mask (nan for land, 1 for ocean) to multiply data by after detrending
landmask = xr.where(ts_hist_anom_load.isel(time=0).isnull()==1, np.nan, 1)

## Detrend
# HISTORICAL
# Need to fill na with a value for detrending
ts_hist_detrend, ts_hist_detrend_resid = xarray_linear_detrend(ts_hist_anom_load.fillna(-999))
# Replace land values that were filled with a value with na
ts_hist_detrend = ts_hist_detrend*landmask

## MCB
ts_mcb_detrend = {}
for key in mcb_keys:
    print(key)
    # ANOMALIES
    # Detrend all other experiments using control residuals
    ts_mcb_detrend[key] =ts_mcb_anom_load[key]-ts_hist_detrend_resid
    # Replace land values that were filled with a value with na
    ts_mcb_detrend[key] = ts_mcb_detrend[key]*landmask
    # print(np.nanmean(ts_mcb_detrend[key]))


#%% CALCULATE E/C INDEX
### PROCESS TIME SERIES FOR EOF ANALYSIS
## Calculate DJF mean for each year for historical and MCB 
# Historical
ts_hist_detrend_djf = djf_mean_annual(ts_hist_detrend).sel(time=slice(None,2017)) # set 2017 as the last complete year with DJF mean
# MCB
ts_mcb_detrend_djf = {}
for key in mcb_keys:
    print(key)
    # Calculate DJF mean for each year
    tmp_mcb_djf = djf_mean_annual(ts_mcb_detrend[key])
    # Append MCB case onto historical control along time dim
    ts_mcb_detrend_djf[key] = xr.concat([ts_hist_detrend_djf,tmp_mcb_djf],dim='time')

# Define number of modes you want to calculate EOFs
nmodes=2

## Historical EOF analysis (DJF ONLY)
# Mask E/C index region
ts_hist_detrend_enso = ts_hist_detrend_djf.where(ecindex_mask>0,drop=True)
# Calculate Niño 3 index time series for historical
ts_hist_detrend_nino3 = fun.calc_weighted_mean_tseries(ts_hist_detrend_djf.where(nino3_mask>0,drop=True))
# Comput EOFs and PCS
ts_hist_eofs, ts_hist_vf, ts_hist_pcs = eof_calc(ts_hist_detrend_enso, nmodes)
# Scale PC1 sign to be positively correlated with Nino3
ts_hist_eofs, ts_hist_pcs = pc_scaling(ts_hist_detrend_djf, ts_hist_detrend_nino3, ts_hist_eofs, ts_hist_pcs, nmodes)
# Calculate E and C indices
pc1 = ts_hist_pcs.sel(mode=0)
pc2 = ts_hist_pcs.sel(mode=1)
ts_hist_eindex = (pc1-pc2)/(np.sqrt(2))
ts_hist_cindex = (pc1+pc2)/(np.sqrt(2))


## MCB EOF analysis (DJF ONLY)
ts_mcb_eofs = {}
ts_mcb_vf = {}
ts_mcb_pcs = {}
ts_mcb_eindex = {}
ts_mcb_cindex = {}
for key in mcb_keys:
    print(key)
    eofs = []
    vfs = []
    pcs = []
    for m in intersect_members:
        # Mask E/C index region
        ts_mcb_detrend_enso = ts_mcb_detrend_djf[key].sel(member=m).where(ecindex_mask>0,drop=True)
        # Calculate Niño 3 index time series for historical
        ts_mcb_detrend_nino3 = fun.calc_weighted_mean_tseries(ts_mcb_detrend_djf[key].sel(member=m).where(nino3_mask>0,drop=True))
        # Compute EOFs and PCS
        eof, vf, pc = eof_calc(ts_mcb_detrend_enso, nmodes)
        # Scale PC1 sign to be positively correlated with Nino3
        eof, pc = pc_scaling(ts_hist_detrend_djf, ts_mcb_detrend_nino3, eof, pc, nmodes)
        eofs.append(eof)
        vfs.append(vf)
        pcs.append(pc)
    # Concatenate along member dimension
    eof_tmp = xr.concat(eofs, dim="member")
    vf_tmp  = xr.concat(vfs, dim="member")
    pc_tmp  = xr.concat(pcs, dim="member")
    # Compute ensemble means
    ts_mcb_eofs[key] = eof_tmp.mean(dim='member')
    ts_mcb_vf[key] = vf_tmp.mean(dim='member')
    ts_mcb_pcs[key] = pc_tmp.mean(dim='member')
    # Calculate E and C indices
    pc1 = ts_mcb_pcs[key].sel(mode=0)
    pc2 = ts_mcb_pcs[key].sel(mode=1)
    ts_mcb_eindex[key] = (pc1-pc2)/(np.sqrt(2))
    ts_mcb_cindex[key] = (pc1+pc2)/(np.sqrt(2))



#%% CALCULATE PMM INDICES
### PROCESS TIME SERIES FOR EOF ANALYSIS
# Concatenate historical time series onto MCB
ts_mcb_detrend_concat = {}
for key in mcb_keys:
    # Append MCB case onto historical control along time dim
    ts_mcb_detrend_concat[key] = xr.concat([ts_hist_detrend,ts_mcb_detrend[key]],dim='time')

# Define number of modes you want to calculate EOFs
nmodes=2

## Historical EOF analysis
# Mask PMM index regions
ts_hist_detrend_npmm = ts_hist_detrend.where(npmm_mask>0,drop=True)
ts_hist_detrend_spmm = ts_hist_detrend.where(spmm_mask>0,drop=True)
# Mask tropical Pacific SSTs
ts_hist_detrend_trop = ts_hist_detrend.where(trop_mask>0,drop=True)
# Calculate Niño 3 index time series for historical
ts_hist_detrend_nino3 = fun.calc_weighted_mean_tseries(ts_hist_detrend.where(nino3_mask>0,drop=True))
# Compute EOFs and PCS of tropical Pacific SSTs
ts_hist_eofs_trop, ts_hist_vf_trop, ts_hist_pcs_trop = eof_calc(ts_hist_detrend_trop, nmodes)
# Regress out tropical Pacific SSTs
ts_hist_detrend_npmm_processed, ts_hist_detrend_npmm_resid = remove_cov(ts_hist_detrend_npmm,ts_hist_pcs_trop.isel(mode=0).values)
ts_hist_detrend_spmm_processed, ts_hist_detrend_spmm_resid = remove_cov(ts_hist_detrend_spmm,ts_hist_pcs_trop.isel(mode=0).values)
# Compute EOFs and PCS of PMMs
ts_hist_eofs_npmm, ts_hist_vf_npmm, ts_hist_pcs_npmm = eof_calc(ts_hist_detrend_npmm_processed, nmodes)
ts_hist_eofs_spmm, ts_hist_vf_spmm, ts_hist_pcs_spmm = eof_calc(ts_hist_detrend_spmm_processed, nmodes)
# Scale PC1 sign to be positively correlated with Nino3
ts_hist_eofs_npmm, ts_hist_pcs_npmm = pc_scaling(ts_hist_detrend_npmm, ts_hist_detrend_nino3, ts_hist_eofs_npmm, ts_hist_pcs_npmm, nmodes)
ts_hist_eofs_spmm, ts_hist_pcs_spmm = pc_scaling(ts_hist_detrend_spmm, ts_hist_detrend_nino3, ts_hist_eofs_spmm, ts_hist_pcs_spmm, nmodes)


## MCB EOF analysis 
ts_mcb_eofs_npmm = {}
ts_mcb_vf_npmm = {}
ts_mcb_pcs_npmm = {}
ts_mcb_eofs_spmm = {}
ts_mcb_vf_spmm = {}
ts_mcb_pcs_spmm = {}
for key in mcb_keys:
    print(key)
    eofs_npmm = []
    vfs_npmm = []
    pcs_npmm = []
    eofs_spmm = []
    vfs_spmm = []
    pcs_spmm = []    
    for m in intersect_members:
        print(m)
        ## Historical EOF analysis
        # Mask PMM index regions
        ts_mcb_detrend_npmm = ts_mcb_detrend_concat[key].sel(member=m).where(npmm_mask>0,drop=True)
        ts_mcb_detrend_spmm = ts_mcb_detrend_concat[key].sel(member=m).where(spmm_mask>0,drop=True)
        # Mask tropical Pacific SSTs
        ts_mcb_detrend_trop = ts_mcb_detrend_concat[key].sel(member=m).where(trop_mask>0,drop=True)
        # Compute EOFs and PCS of tropical Pacific SSTs
        ts_mcb_eofs_trop, ts_mcb_vf_trop, ts_mcb_pcs_trop = eof_calc(ts_mcb_detrend_trop, nmodes)
        # Regress out tropical Pacific SSTs
        ts_mcb_detrend_npmm_processed, ts_mcb_detrend_npmm_resid = remove_cov(ts_mcb_detrend_npmm,ts_mcb_pcs_trop.isel(mode=0).values)
        ts_mcb_detrend_spmm_processed, ts_mcb_detrend_spmm_resid = remove_cov(ts_mcb_detrend_spmm,ts_mcb_pcs_trop.isel(mode=0).values)
        # Compute EOFs and PCS of PMMs
        eof_npmm, vf_npmm, pc_npmm = eof_calc(ts_mcb_detrend_npmm_processed, nmodes)
        eof_spmm, vf_spmm, pc_spmm = eof_calc(ts_mcb_detrend_spmm_processed, nmodes)
        # Scale PC1 sign to be positively correlated with Nino3
        eof_npmm, pc_npmm = pc_scaling(ts_hist_detrend_npmm, ts_hist_detrend_nino3, eof_npmm, pc_npmm, nmodes)
        eof_spmm, pc_spmm = pc_scaling(ts_hist_detrend_spmm, ts_hist_detrend_nino3, eof_spmm, pc_spmm, nmodes)
        # Append members
        eofs_npmm.append(eof_npmm)
        vfs_npmm.append(vf_npmm)
        pcs_npmm.append(pc_npmm)
        eofs_spmm.append(eof_spmm)
        vfs_spmm.append(vf_spmm)
        pcs_spmm.append(pc_spmm)        
    # Concatenate along member dimension
    eof_tmp_npmm = xr.concat(eofs_npmm, dim="member")
    vf_tmp_npmm  = xr.concat(vfs_npmm, dim="member")
    pc_tmp_npmm  = xr.concat(pcs_npmm, dim="member")
    eof_tmp_spmm = xr.concat(eofs_spmm, dim="member")
    vf_tmp_spmm  = xr.concat(vfs_spmm, dim="member")
    pc_tmp_spmm  = xr.concat(pcs_spmm, dim="member")    
    # Compute ensemble means
    ts_mcb_eofs_npmm[key] = eof_tmp_npmm
    ts_mcb_vf_npmm[key] = vf_tmp_npmm
    ts_mcb_pcs_npmm[key] = pc_tmp_npmm
    ts_mcb_eofs_spmm[key] = eof_tmp_spmm
    ts_mcb_vf_spmm[key] = vf_tmp_spmm
    ts_mcb_pcs_spmm[key] = pc_tmp_spmm


## CALCULATE GLOBAL SSTs w/ ENSO VARIABILITY REMOVED
# Check residuals after tropical Pacific EOF1 removal
# Mask tropical Pacific SSTs
ts_hist_detrend_trop = ts_hist_detrend.where(trop_mask>0,drop=True)
# Compute EOFs and PCS of tropical Pacific SSTs
ts_hist_eofs_trop, ts_hist_vf_trop, ts_hist_pcs_trop = eof_calc(ts_hist_detrend_trop, nmodes)
# Regress out tropical Pacific SSTs
ts_hist_detrend_processed, ts_hist_detrend_resid = remove_cov(ts_hist_detrend,ts_hist_pcs_trop.isel(mode=0).values)

key_subset = ['06-02', '06-08']

mcb_processed_xr = {}
mcb_resid_xr = {}

for key in key_subset:
    print(key)
    mcb_processed = []
    mcb_resid = []
    for m in intersect_members:
        print(m)
        ## Historical EOF analysis
        # Mask tropical Pacific SSTs
        ts_mcb_detrend_trop = ts_mcb_detrend_concat[key].sel(member=m).where(trop_mask>0,drop=True)
        # Compute EOFs and PCS of tropical Pacific SSTs
        ts_mcb_eofs_trop, ts_mcb_vf_trop, ts_mcb_pcs_trop = eof_calc(ts_mcb_detrend_trop, nmodes)
        # Regress out tropical Pacific SSTs
        ts_mcb_detrend_proccessed, ts_mcb_detrend_resid = remove_cov(ts_mcb_detrend_concat[key].sel(member=m),ts_mcb_pcs_trop.isel(mode=0).values)
        # Append members
        mcb_processed.append(ts_mcb_detrend_proccessed)
        mcb_resid.append(ts_mcb_detrend_resid)
    # Concatenate along member dimension
    mcb_processed_xr[key] = xr.concat(mcb_processed, dim='member')
    mcb_resid_xr[key] = xr.concat(mcb_resid, dim='member')


# Calculate standard error of control ensemble for DJF of ENSO event
p_val_xr = {}
for key in key_subset:
    print(key)
    # Select matching time period (e.g., DJF of target El Niño year)
    mcb_djf =mcb_processed_xr[key].isel(time=slice(-24,None)).isel(time=slice(4,16)) 
    mcb_djf=mcb_djf.loc[{'time':[t for t in pd.to_datetime(mcb_djf.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    if yr=='2015':
        hist_djf = ts_hist_detrend_processed.isel(time=slice(-30,None)).isel(time=slice(4,16))  
    elif yr=='1997':
        hist_djf = ts_hist_detrend_processed.isel(time=slice(-246,-246+30)).isel(time=slice(4,16)) 
    hist_djf=hist_djf.loc[{'time':[t for t in pd.to_datetime(hist_djf.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    # Flatten 
    mcb_vals = mcb_djf.values 
    hist_vals = hist_djf.values 
    # Reshape
    mcb_flat = mcb_vals.reshape(-1, *mcb_vals.shape[-2:])  
    t_stat, p_val = stats.ttest_ind(mcb_flat, hist_vals, axis=0, equal_var=False)  # Welch's t-test
    p_val_xr[key] = xr.DataArray(p_val, coords={'lat': hist_djf.lat, 'lon': hist_djf.lon}, dims=['lat', 'lon'])


# MCB Full Effort - Control
subplot_num=0
if year_init=='2015':
    subplot_lab = 'A'
elif year_init=='1997':
    subplot_lab = 'B'
exp_name = {'06-02': 'Full effort'}
cmin=-3
cmax=3

fig = plt.figure(figsize=(5,4));
for key in key_subset[:-1]:
    anom = fun.reorient_netCDF(djf_mean_annual(mcb_processed_xr[key].isel(time=slice(-24,None)).mean(dim='member')).sel(time=int(yr)+1), target=360) - fun.reorient_netCDF(djf_mean_annual(ts_hist_detrend_processed).sel(time=int(yr)+1), target=360)
    p_val_reoriented = fun.reorient_netCDF(p_val_xr[key], target=360)
    insig_mask = (p_val_reoriented >= 0.05).values  # True where NOT significant
    yi, xi = np.where(insig_mask)  # row/col indices for imshow
    mcb, p1 = fun.plot_panel_maps(in_xr=anom, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                                CI_in=insig_mask,CI_level=0.05,CI_display='inv_stipple',\
                                projection='Robinson',nrow=1,ncol=1,subplot_num=subplot_num,mean_val='none',cbar=False, continent_fill=True)
    plt.annotate(yr+'-'+str(int(yr)+1)+' '+exp_name[key]+' MCB (ENSO removed)', xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
    plt.title(subplot_lab, fontweight='bold',fontsize=14, loc='left')
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);
## Add colorbars to bottom of figure
cbar_ax = fig.add_axes([0.2, 0.2, 0.6, 0.04]) #rect kwargs [left, bottom, width, height];
plt.colorbar(p1, cax = cbar_ax, orientation='horizontal', label='Temperature (\N{DEGREE SIGN}C)', extend='both',pad=0.1);