### PURPOSE: Script to define ENSO temperature and precipitation regional impacts
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/05/2026
### NOTES: adapted from smyle_mcb_eof_v5.py

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

plt.ion();

dask.config.set({"array.slicing.split_large_chunks": False})

#run this line in console before starting ipython if getting OMP_NUM_THREADS error
os.environ["OMP_NUM_THREADS"] = "1"

##################################################################################################################
## WHICH EXPERIMENT ARE YOU READING IN? ##
month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
year_init = input('Which initialization year are you reading in (1997, 2015?): ')
enso_phase = 'nino'
sensitivity_opt = 'y'
## UNCOMMENT THESE OPTIONS FOR DEMO ##
year_init = '2015'
month_init = '05'
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
atm_varnames_monthly_subset = ['TS','PRECT','LANDFRAC']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day


## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS
# Read in each ensemble member as a discontinuous time series by concatenating overlapping periods
atm_monthly_ctrl_clim_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*atm_drift_clim.v3.nc')[0])))


## COMPUTE LONG TERM STANDARD DEVIATION AND MONTHLY CLIMATOLOGY MEAN
# Subset time from 1970-2014
hist_ext = atm_monthly_ctrl_clim_xr.isel(member=slice(0,len(intersect_members)))[['TS','PRECT']]
# Calculate monthly climatological mean
hist_clim_ens_mean = hist_ext.mean(dim=('member')).groupby('time.month').mean()
# Calculate standard deviation
hist_ens_sd = hist_ext.std(dim=('time','member'))


# Create formatted historical time series to append to control and MCB runs
hist_window = atm_monthly_ctrl_clim_xr.isel(time=atm_monthly_ctrl_clim_xr['time.year']<=2017).isel(member=slice(0,len(intersect_members)))
# Reassign ensemble member label so it can be appended to control and MCB runs
hist_window = hist_window.assign_coords(member=intersect_members)


# CALCULATE MONTHLY ANOMALIES FOR SMYLE HISTORICAL (1970-2017)
ts_hist_anom = hist_window.TS.groupby('time.month') - hist_clim_ens_mean.TS
ts_hist_anom.attrs['units'] = '\N{DEGREE SIGN}C'
prect_hist_anom = hist_window.PRECT.groupby('time.month') - hist_clim_ens_mean.PRECT
prect_hist_anom.attrs['units'] = 'mm/day'


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
    ## PRECT
    # # Convert from m/s to mm/day
    m_to_mm = 1e3 #mm/m
    s_to_days = 86400 #s/day
    atm_monthly_ctrl[key] = atm_monthly_ctrl[key].assign(PRECT=atm_monthly_ctrl[key]['PRECT']*m_to_mm*s_to_days)
    atm_monthly_ctrl[key]['PRECT'].attrs['units'] = 'mm/day'
    ## TS
    # Convert from K to C
    atm_monthly_ctrl[key] = atm_monthly_ctrl[key].assign(TS=atm_monthly_ctrl[key]['TS']-273.15)
    atm_monthly_ctrl[key]['TS'].attrs['units'] = '°C'
    # Compute climatological anomalies
    i_month=np.arange(1,13,1)
    ts_ctrl_copy = atm_monthly_ctrl[key]['TS']*1
    prect_ctrl_copy = atm_monthly_ctrl[key]['PRECT']*1
    ## TS
    ts_ctrl_anom[key] = ts_ctrl_copy.groupby('time.month') - hist_clim_ens_mean.TS
    ## PRECT
    prect_ctrl_anom[key] = prect_ctrl_copy.groupby('time.month') - hist_clim_ens_mean.PRECT
    # Reassign units
    ## TS
    ts_ctrl_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    ## PRECT
    prect_ctrl_anom[key].attrs['units']='mm/day'



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
    ## PRECT
    # # Convert from m/s to mm/day
    m_to_mm = 1e3 #mm/m
    s_to_days = 86400 #s/day
    atm_monthly_mcb[key] = atm_monthly_mcb[key].assign(PRECT=atm_monthly_mcb[key]['PRECT']*m_to_mm*s_to_days)
    atm_monthly_mcb[key]['PRECT'].attrs['units'] = 'mm/day'
    ## TS
    # Convert from K to C
    atm_monthly_mcb[key] = atm_monthly_mcb[key].assign(TS=atm_monthly_mcb[key]['TS']-273.15)
    atm_monthly_mcb[key]['TS'].attrs['units'] = '°C'
    # Compute climatological anomalies
    i_month=np.arange(1,13,1)
    ts_mcb_copy = atm_monthly_mcb[key]['TS']*1
    prect_mcb_copy = atm_monthly_mcb[key]['PRECT']*1 
    ## TS
    ts_mcb_anom[key] = ts_mcb_copy.groupby('time.month') - hist_clim_ens_mean.TS
    ## PRECT
    prect_mcb_anom[key] = prect_mcb_copy.groupby('time.month') - hist_clim_ens_mean.PRECT
    # Reassign units
    ## TS
    ts_mcb_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    ## PRECT
    prect_mcb_anom[key].attrs['units']='mm/day'


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


# Read in LENS2 SDs
data_dir = '/_data/LENS2/'
lens2_ts_sd = fun.reorient_netCDF(xr.open_dataset(data_dir+'CESM-LENS2.hist.ensemble.std.TS.monthly_clim.1970-2014.nc')).TS
lens2_prect_sd = fun.reorient_netCDF(xr.open_dataset(data_dir+'CESM-LENS2.hist.ensemble.std.PRECT.monthly_clim.1970-2014.nc')).PRECT
# Align grid with CESM grid (fix rounding errors from regridding)
lens2_ts_sd = lens2_ts_sd.assign_coords({'lon':ts_ctrl_anom[''].lon,\
                            'lat':ts_ctrl_anom[''].lat})
lens2_prect_sd = lens2_prect_sd.assign_coords({'lon':ts_ctrl_anom[''].lon,\
                            'lat':ts_ctrl_anom[''].lat})
# Turn ocean values to nan
landmask = atm_monthly_ctrl['']['LANDFRAC'].isel(time=0,member=0).load()


### DETREND ANOMALIES AND LANDMASKED SD NORMALIZED ANOMALIES
## Load in all anomaly files at once (required for detrending) and convert to standard deviation space
# CONTROL
ts_ctrl_anom_load = {}
prect_ctrl_anom_load = {}
for key in ctrl_keys:
    ts_ctrl_anom_load[key] = ts_ctrl_anom[key].load().groupby('time.month')/lens2_ts_sd
    prect_ctrl_anom_load[key] = prect_ctrl_anom[key].load().groupby('time.month')/lens2_prect_sd

# MCB
ts_mcb_anom_load = {}
prect_mcb_anom_load = {}
for key in mcb_keys:
    print(key)
    ts_mcb_anom_load[key] = ts_mcb_anom[key].load().groupby('time.month')/lens2_ts_sd
    prect_mcb_anom_load[key] = prect_mcb_anom[key].load().groupby('time.month')/lens2_prect_sd

# HISTORICAL
ts_hist_anom_load = ts_hist_anom.load().groupby('time.month')/lens2_ts_sd
prect_hist_anom_load = prect_hist_anom.load().groupby('time.month')/lens2_prect_sd


## Detrend with the historical predicted values
# HISTORICAL
ts_hist_detrend, ts_hist_detrend_resid = fun.xarray_linear_detrend(ts_hist_anom_load.mean(dim='member'))
prect_hist_detrend, prect_hist_detrend_resid = fun.xarray_linear_detrend(prect_hist_anom_load.mean(dim='member'))
# Mask out land
ts_hist_detrend_sd_land= xr.where((landmask>0.1),ts_hist_detrend,np.nan)
prect_hist_detrend_sd_land= xr.where((landmask>0.1),prect_hist_detrend,np.nan)


# Create continuous time series by averaging over overlapping periods for El Niño years
tcont = np.unique(ts_ctrl_anom_load[''].time.values)
ts_hist_detrend_resid_subset = ts_ctrl_anom_load[''].isel(member=0)*np.nan
prect_hist_detrend_resid_subset = prect_ctrl_anom_load[''].isel(member=0)*np.nan
for t in tcont:
    t_unique = ts_hist_detrend_resid.where((ts_hist_detrend_resid.time==t),drop=True).mean(dim='time')
    p_unique = prect_hist_detrend_resid.where((prect_hist_detrend_resid.time==t),drop=True).mean(dim='time')
    ts_hist_detrend_resid_subset = xr.where((ts_hist_detrend_resid_subset.time==t),t_unique,ts_hist_detrend_resid_subset)
    prect_hist_detrend_resid_subset = xr.where((prect_hist_detrend_resid_subset.time==t),p_unique,prect_hist_detrend_resid_subset)


# CONTROL
ts_ctrl_detrend_land_sd = {}
prect_ctrl_detrend_land_sd = {}
for key in ctrl_keys:
    # Mask out land
    ts_sd_land_only = xr.where((landmask>0.1),ts_ctrl_anom_load[key],np.nan)
    prect_sd_land_only = xr.where((landmask>0.1),prect_ctrl_anom_load[key],np.nan)
    # Detrend with historical values
    ts_ctrl_detrend_land_sd[key] = ts_sd_land_only-ts_hist_detrend_resid_subset
    prect_ctrl_detrend_land_sd[key] = prect_sd_land_only-prect_hist_detrend_resid_subset

## MCB
ts_mcb_detrend_land_sd = {}
prect_mcb_detrend_land_sd = {}
for key in mcb_keys:
    print(key)
    # Mask out land
    ts_sd_land_only = xr.where((landmask>0.1),ts_mcb_anom_load[key],np.nan)
    prect_sd_land_only = xr.where((landmask>0.1),prect_mcb_anom_load[key],np.nan)
    # Detrend with historical values
    ts_mcb_detrend_land_sd[key] = ts_sd_land_only-ts_hist_detrend_resid_subset
    prect_mcb_detrend_land_sd[key] = prect_sd_land_only-prect_hist_detrend_resid_subset



#%% CALCULATE T AND P ANOMALIES FOR 9 MONTH MEAN
# Create El Niño DJF average
if year_init=='1997':
    peak_yrs = [1997,1998]
elif year_init=='2015':
    peak_yrs = [2015,2016]
# Create June-August subset (15 months) for peak ENSO year
def june_feb_subset(data):
    xr_subset = data.where(((data['time.year']==peak_yrs[0])&(data['time.month']>=6))|\
                                        ((data['time.year']==peak_yrs[1])&(data['time.month']<=2))\
                                        ,drop=True)
    return xr_subset

# Subset Jun-Aug s.d. normalized anomalies
ts_ctrl_detrend_jun_aug = {}
ts_mcb_detrend_jun_aug = {}
ts_mcb_anom_jun_aug = {}

prect_ctrl_detrend_jun_aug = {}
prect_mcb_detrend_jun_aug = {}
prect_mcb_anom_jun_aug = {}

sum_ctrl_detrend_jun_aug = {}
sum_mcb_detrend_jun_aug = {}
sum_mcb_anom_jun_aug = {}

# CONTROL
for key in ctrl_keys:
    # Subset June t-1 to Feb t and take mean
    ts_ctrl_detrend_jun_aug[key] = fun.weighted_temporal_mean(june_feb_subset(ts_ctrl_detrend_land_sd[key]), by_year=False)
    prect_ctrl_detrend_jun_aug[key] = fun.weighted_temporal_mean(june_feb_subset(prect_ctrl_detrend_land_sd[key]), by_year=False)
    # Calculate absolute anomaly sum
    sum_ctrl_detrend_jun_aug[key] = np.abs(ts_ctrl_detrend_jun_aug[key]) + np.abs(prect_ctrl_detrend_jun_aug[key])
    # Create anomaly mask based on standard deviation threshold for T and P
    sd_max = 2 ## USER INPUT
    ts_warm_mask = xr.where(ts_ctrl_detrend_jun_aug[key].mean(dim='member')>sd_max, 1, 0)
    ts_cool_mask = xr.where(ts_ctrl_detrend_jun_aug[key].mean(dim='member')<-sd_max, 1, 0)
    prect_wet_mask = xr.where(prect_ctrl_detrend_jun_aug[key].mean(dim='member')>sd_max, 1, 0)
    prect_dry_mask = xr.where(prect_ctrl_detrend_jun_aug[key].mean(dim='member')<-sd_max, 1, 0)

# MCB
for key in mcb_keys:
    print(key)
    # Subset June t-1 to Feb t and take mean
    ts_mcb_detrend_jun_aug[key] = fun.weighted_temporal_mean(june_feb_subset(ts_mcb_detrend_land_sd[key]), by_year=False)
    prect_mcb_detrend_jun_aug[key] = fun.weighted_temporal_mean(june_feb_subset(prect_mcb_detrend_land_sd[key]), by_year=False)
    # Calculate absolute anomaly sum
    sum_mcb_detrend_jun_aug[key] = np.abs(ts_mcb_detrend_jun_aug[key]) + np.abs(prect_mcb_detrend_jun_aug[key])
    # Calculate MCB anomalies from control
    ts_mcb_anom_jun_aug[key] =  ts_mcb_detrend_jun_aug[key] - ts_ctrl_detrend_jun_aug['']
    prect_mcb_anom_jun_aug[key] =  prect_mcb_detrend_jun_aug[key] - prect_ctrl_detrend_jun_aug['']
    sum_mcb_anom_jun_aug[key] =  sum_mcb_detrend_jun_aug[key] - sum_ctrl_detrend_jun_aug['']


# Create region specific masks based on SD thresholds and lat/lon bounds
# Define lat/lon variables
lat = ts_ctrl_detrend_jun_aug[''].lat
lon = ts_ctrl_detrend_jun_aug[''].lon

# a) Asia warming
lat_min = -11
lat_max = 60
lon_min = 66
lon_max = 150
asia_warm = xr.where((ts_warm_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
asia_warm = asia_warm.rename('ASIA')

# b) Western North America warming
lat_min = 20
lat_max = 90
lon_min = -170
lon_max = -90
wna_warm = xr.where((ts_warm_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
wna_warm = wna_warm.rename('W. NAM')

# c) South America warming
lat_min = -60
lat_max = 20
lon_min = -100
lon_max = -30
wsa_warm = xr.where((ts_warm_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
wsa_warm = wsa_warm.rename('SAM')

# d) Africa warming
lat_min = -60
lat_max = 25
lon_min = -25
lon_max = 60
saf_warm = xr.where((ts_warm_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
saf_warm = saf_warm.rename('AFR')

# e) Australia warming
lat_min = -40
lat_max = -11
lon_min = 112
lon_max = 154
aus_warm = xr.where((ts_warm_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
aus_warm = aus_warm.rename('AUS')

# f) Eastern North America cooling
lat_min = 20
lat_max = 50
lon_min = -150
lon_max = -75
ena_cool = xr.where((ts_cool_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
ena_cool = ena_cool.rename('E. NAM')

# g) Southern South America cooling
lat_min = -60
lat_max = -20
lon_min = -90
lon_max = -30
ssa_cool = xr.where((ts_cool_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
ssa_cool = ssa_cool.rename('S. SAM')

# h) Europe cooling
lat_min = 35
lat_max = 90
lon_min = -15
lon_max = 40
eur_cool = xr.where((ts_cool_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
eur_cool = eur_cool.rename('EUR')

# i) East Africa cooling
lat_min = -5
lat_max = 15
lon_min = 35
lon_max = 52
eaf_cool = xr.where((ts_cool_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
eaf_cool = eaf_cool.rename('E. AFR')

# j) SE Asia cooling
lat_min = -15
lat_max = 20
lon_min = 73
lon_max = 180
asia_cool = xr.where((ts_cool_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
asia_cool = asia_cool.rename('S.E. ASIA')

# n) Southern South America wettening
lat_min = -60
lat_max = -20
lon_min = -90
lon_max = -30
ssa_wet = xr.where((prect_wet_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
ssa_wet = ssa_wet.rename('SAM')

# o) Eastern Africa wettening
lat_min = -20
lat_max = 15
lon_min = 0
lon_max = 52
ceaf_wet = xr.where((prect_wet_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
ceaf_wet = ceaf_wet.rename('E. AFR')

# p) Middle East wettening
lat_min = 10
lat_max = 55
lon_min = 25
lon_max = 90
mie_wet = xr.where((prect_wet_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
mie_wet = mie_wet.rename('ME')

# q) Central/South America drying
lat_min = -10
lat_max = 30
lon_min = -120
lon_max = -30
csa_dry = xr.where((prect_dry_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
csa_dry = csa_dry.rename('CAM/SAM')

# r) Western Africa drying
lat_min = -10
lat_max = 30
lon_min = -25
lon_max = 10
waf_dry = xr.where((prect_dry_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
waf_dry = waf_dry.rename('W. AFR')

# s) Asia drying
lat_min = -11
lat_max = 60
lon_min = 66
lon_max = 150
asia_dry = xr.where((prect_dry_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
asia_dry = asia_dry.rename('ASIA')

# t) Australia drying
lat_min = -45
lat_max = -11
lon_min = 112
lon_max = 154
aus_dry = xr.where((prect_dry_mask>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
aus_dry = aus_dry.rename('AUS')

# Combine warm, cold, wet, and dry regions into four xarrays
warm_regions_combined = xr.merge([asia_warm,wna_warm,wsa_warm,saf_warm,aus_warm])
cool_regions_combined = xr.merge([ena_cool,ssa_cool,eur_cool,eaf_cool,asia_cool])
wet_regions_combined = xr.merge([ssa_wet,ceaf_wet,mie_wet])
dry_regions_combined = xr.merge([csa_dry,waf_dry,asia_dry,aus_dry])


# Subset MCB anomalies by region and save output as dataframe for plotting
mcb_keys_sub = ['06-02','06-08','12-02']
metrics_keys = ['Warm','Cool','Wet','Dry']
regional_anomalies_df = pd.DataFrame()
for key in mcb_keys_sub:
    print(key)
    for metric in metrics_keys:
        if metric=='Warm':
            xr_in = ts_mcb_anom_jun_aug[key]
            xr_mask = warm_regions_combined
        elif metric=='Cool':
            xr_in = ts_mcb_anom_jun_aug[key]
            xr_mask = cool_regions_combined
        elif metric=='Wet':
            xr_in = prect_mcb_anom_jun_aug[key]
            xr_mask = wet_regions_combined
        elif metric=='Dry':
            xr_in = prect_mcb_anom_jun_aug[key]
            xr_mask = dry_regions_combined        
        regions = list(xr_mask.keys())
        for region in regions:
            anom = float(fun.calc_weighted_mean_tseries(xr.where(xr_mask[region]>0,xr_in,np.nan)).mean())
            se = float(fun.calc_weighted_mean_tseries(xr.where(xr_mask[region]>0,xr_in,np.nan)).std()/np.sqrt(len(xr_in.member)))
            new_row = pd.Series({'experiment':key, 'metric':metric, 'region':region, 'anom':anom,'se':se})
            regional_anomalies_df = pd.concat([regional_anomalies_df,new_row.to_frame().T],ignore_index=True)


## FIGURE 4. T and P anomaly maps with contoured regions + bar plots
# Set figure dimensions and grid
fig = plt.figure(figsize=(10, 8))
spec = fig.add_gridspec(4, 4)
# A) Temperature SD
plot_proj = ccrs.Robinson(central_longitude=0)
ax0 = fig.add_subplot(spec[0:2, 0:2], projection=plot_proj,transform=plot_proj)
p=ts_ctrl_detrend_jun_aug[''].mean(dim='member').plot(ax=ax0,vmin=-10,vmax=10,cmap='RdBu_r',transform=ccrs.PlateCarree(), add_labels=False,add_colorbar=True,cbar_kwargs={'orientation':'horizontal','extend':'both','shrink':0.6,'label':'Temperature (s.d.)'})
ax0.set_title('A',fontsize=14,fontweight='bold',loc='left');
ax0.coastlines(color='grey'); p.axes.set_global();
# Add bar plot region contours
# Warm
x = warm_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors='#b2182b', linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=False,add_colorbar=False);
# Cool
x = cool_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors='#2166ac', linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=False,add_colorbar=False);


# B) Precipitation SD
ax0 = fig.add_subplot(spec[2:4, 0:2], projection=plot_proj,transform=plot_proj)
p=prect_ctrl_detrend_jun_aug[''].mean(dim='member').plot(ax=ax0,vmin=-10,vmax=10,cmap='BrBG', transform=ccrs.PlateCarree(), add_labels=False,add_colorbar=True,cbar_kwargs={'orientation':'horizontal','extend':'both','shrink':0.6,'label':'Precipitation (s.d.)'})
ax0.coastlines(color='grey'); p.axes.set_global();
ax0.set_title('B',fontsize=14,fontweight='bold',loc='left');
# Add bar plot region contours
# Wet
x = wet_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors='#018571', linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=False,add_colorbar=False);
# Dry
x = dry_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors='#a6611a', linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=False,add_colorbar=False);

# C) Temperature bar plots
ax1 = fig.add_subplot(spec[0:2, 2:4]);
mcb_legend_longname = {'06-02':'Full effort','06-08':'Early action','06-11':'','09-02':'','09-11':'','12-02':'11th hour'}
bar_colors = ['#66c2a5','#fc8d62','#8da0cb']
bar_hatch = ('....','///','+++')
df_warm = regional_anomalies_df[regional_anomalies_df['metric']=='Warm'].sort_values('region')
df_cool = regional_anomalies_df[regional_anomalies_df['metric']=='Cool'].sort_values('region')
df_in =pd.concat([df_warm,df_cool],ignore_index=True)
pivot_mean = df_in[['experiment','region','anom','metric']].pivot_table(index='region',columns='experiment',sort=False)
pivot_se = df_in[['experiment','region','se','metric']].pivot_table(index='region',columns='experiment',sort=False)
bar_width = 0.25
x = np.arange(len(pivot_mean.index))
for i, column in enumerate(pivot_mean.columns):
    if i==0:
        bars = ax1.barh(x + i * bar_width, pivot_mean[column],bar_width,
                    label=column, color='k',edgecolor='k',xerr=2*pivot_se[('se',column[1])], error_kw={'ecolor':'grey','elinewidth': 1, 'capsize':0})
    elif i==len(pivot_mean.columns)-1:
        bars = ax1.barh(x + i * bar_width, pivot_mean[column], bar_width,
                    label=column, fill=None, edgecolor='k',xerr=2*pivot_se[('se',column[1])], error_kw={'ecolor':'grey','elinewidth': 1, 'capsize':0})
    else:
        bars = ax1.barh(x + i * bar_width, pivot_mean[column], bar_width,
                    label=column, fill=None, edgecolor='k',hatch='///',xerr=2*pivot_se[('se',column[1])], error_kw={'ecolor':'grey','elinewidth': 1, 'capsize':0})
ax1.set_yticks(x + bar_width / 3)
ax1.set_yticklabels(pivot_mean.index)
plt.legend(['Full effort', 'Early action','11th hour'],loc='lower left')
ax1.invert_yaxis();
plt.xlim(-6,6);
plt.axvline(0,ymin=ax1.get_ylim()[0],ymax=ax1.get_ylim()[1],color='k',linestyle='dashed');
# Add warm rectangle
w = ax1.get_xlim()[1]-ax1.get_xlim()[0]
h = abs((ax1.get_ylim()[1]-ax1.get_ylim()[0])/len(pivot_mean)*len(np.unique(df_warm['region'])))
ax1.add_patch(Rectangle((ax1.get_xlim()[0],ax1.get_ylim()[1]),width=w,height=h,facecolor='#b2182b',alpha=0.2,zorder=-1))
ax1.annotate('Warm', xy=(5.7,ax1.get_ylim()[1]+h*.2),ha='right',fontsize=12,fontweight='bold',color='#b2182b')
# Add cool rectangle
w = ax1.get_xlim()[1]-ax1.get_xlim()[0]
h = (ax1.get_ylim()[1]-ax1.get_ylim()[0])/len(pivot_mean)*len(np.unique(df_cool['region']))
ax1.add_patch(Rectangle((ax1.get_xlim()[0],ax1.get_ylim()[0]),width=w,height=h,facecolor='#2166ac',alpha=0.2,zorder=-1))
ax1.annotate('Cool', xy=(5.7,ax1.get_ylim()[0]+h*.8),ha='right',fontsize=12,fontweight='bold',color='#2166ac')
# Add labels
ax1.get_legend().remove()
plt.xlabel('\N{GREEK CAPITAL LETTER DELTA} Temperature (s.d.)',fontsize=12);
plt.ylabel('');
ax1.annotate(year_init+'-'+str(int(year_init)+1)+' El Niño', xy=(.5,1.02), fontsize=12, ha='center',xycoords='axes fraction',color='k');
if year_init=='2015':
    ax1.set_title('C',fontsize=14,fontweight='bold',loc='left');
elif year_init=='1997':
    ax1.set_title('E',fontsize=14,fontweight='bold',loc='left');


# D) Precipitation bar plots
ax1 = fig.add_subplot(spec[2:4, 2:4]);
mcb_legend_longname = {'06-02':'Full effort','06-08':'Early action','06-11':'','09-02':'','09-11':'','12-02':'11th hour'}
bar_colors = ['#66c2a5','#fc8d62','#8da0cb']
df_wet = regional_anomalies_df[regional_anomalies_df['metric']=='Wet'].sort_values('region')
df_dry = regional_anomalies_df[regional_anomalies_df['metric']=='Dry'].sort_values('region')
df_in =pd.concat([df_wet,df_dry],ignore_index=True)
pivot_mean = df_in[['experiment','region','anom','metric']].pivot_table(index='region',columns='experiment',sort=False)
pivot_se = df_in[['experiment','region','se','metric']].pivot_table(index='region',columns='experiment',sort=False)
bar_width = 0.25
x = np.arange(len(pivot_mean.index))
for i, column in enumerate(pivot_mean.columns):
    if i==0:
        bars = ax1.barh(x + i * bar_width, pivot_mean[column],bar_width,
                    label=column, color='k',edgecolor='k',xerr=2*pivot_se[('se',column[1])], error_kw={'ecolor':'grey','elinewidth': 1, 'capsize':0})
    elif i==len(pivot_mean.columns)-1:
        bars = ax1.barh(x + i * bar_width, pivot_mean[column], bar_width,
                    label=column, fill=None, edgecolor='k',xerr=2*pivot_se[('se',column[1])], error_kw={'ecolor':'grey','elinewidth': 1, 'capsize':0})
    else:
        bars = ax1.barh(x + i * bar_width, pivot_mean[column], bar_width,
                    label=column, fill=None, edgecolor='k',hatch='///',xerr=2*pivot_se[('se',column[1])], error_kw={'ecolor':'grey','elinewidth': 1, 'capsize':0})

ax1.set_yticks(x + bar_width / 3)
ax1.set_yticklabels(pivot_mean.index)
if year_init=='1997':
    plt.legend(['Full effort', 'Early action','11th hour'],loc='lower left');
ax1.invert_yaxis();
plt.xlim(-3,3);
plt.axvline(0,ymin=ax1.get_ylim()[0],ymax=ax1.get_ylim()[1],color='k',linestyle='dashed');
# Add wet rectangle
w = ax1.get_xlim()[1]-ax1.get_xlim()[0]
h = abs((ax1.get_ylim()[1]-ax1.get_ylim()[0])/len(pivot_mean)*len(np.unique(df_wet['region'])))
ax1.add_patch(Rectangle((ax1.get_xlim()[0],ax1.get_ylim()[1]),width=w,height=h,facecolor='#018571',alpha=0.2,zorder=-1))
ax1.annotate('Wet', xy=(2.7,ax1.get_ylim()[1]+h*.2),ha='right',fontsize=12,fontweight='bold',color='#018571')
# Add dry rectangle
w = ax1.get_xlim()[1]-ax1.get_xlim()[0]
h = (ax1.get_ylim()[1]-ax1.get_ylim()[0])/len(pivot_mean)*len(np.unique(df_dry['region']))
ax1.add_patch(Rectangle((ax1.get_xlim()[0],ax1.get_ylim()[0]),width=w,height=h,facecolor='#a6611a',alpha=0.2,zorder=-1))
ax1.annotate('Dry', xy=(2.7,ax1.get_ylim()[0]+h*.8),ha='right',fontsize=12,fontweight='bold',color='#a6611a')
# Add labels
plt.xlabel('\N{GREEK CAPITAL LETTER DELTA} Precipitation (s.d.)',fontsize=12);
plt.ylabel('');
if year_init=='2015':
    ax1.set_title('D',fontsize=14,fontweight='bold',loc='left');
elif year_init=='1997':
    ax1.set_title('F',fontsize=14,fontweight='bold',loc='left');

# Wait for graphics to load
plt.tight_layout();



## SI W/ FIG 4. REGIONS LABELED
region_colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
# Set figure dimensions and grid
fig = plt.figure(figsize=(10, 8))
spec = fig.add_gridspec(4, 4)
# a) Temperature SD
plot_proj = ccrs.Robinson(central_longitude=0)
ax0 = fig.add_subplot(spec[0:2, 0:2], projection=plot_proj,transform=plot_proj)
ax0.set_title('A',fontsize=14,fontweight='bold',loc='left');
ax0.coastlines(color='grey'); 
# Warm
subplot_num=0
patch=[]
x = warm_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors=region_colors[subplot_num], linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=region);
    patch.append(mpatches.Patch(color=region_colors[subplot_num], label=region))
    subplot_num+=1
plt.legend(loc='upper center',handles=patch,ncol=2,bbox_to_anchor=(0.5,0));
plt.annotate('Warm', xy=(0.5,1.05), ha='center', xycoords='axes fraction',color='k',fontsize=12);
# Cool
ax0 = fig.add_subplot(spec[0:2, 2:4], projection=plot_proj,transform=plot_proj)
ax0.set_title('B',fontsize=14,fontweight='bold',loc='left');
ax0.coastlines(color='grey'); 
subplot_num=0
patch=[]
x = cool_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors=region_colors[subplot_num], linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=region);
    patch.append(mpatches.Patch(color=region_colors[subplot_num], label=region))
    subplot_num+=1
plt.legend(loc='upper center',handles=patch,ncol=2,bbox_to_anchor=(0.5,0));
plt.annotate('Cool', xy=(0.5,1.05), ha='center', xycoords='axes fraction',color='k',fontsize=12);
# Wet
ax0 = fig.add_subplot(spec[2:4,0:2], projection=plot_proj,transform=plot_proj)
ax0.set_title('C',fontsize=14,fontweight='bold',loc='left');
ax0.coastlines(color='grey'); 
subplot_num=0
patch=[]
x = wet_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors=region_colors[subplot_num], linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=region);
    patch.append(mpatches.Patch(color=region_colors[subplot_num], label=region))
    subplot_num+=1
plt.legend(loc='upper center',handles=patch,ncol=2,bbox_to_anchor=(0.5,0));
plt.annotate('Wet', xy=(0.5,1.05), ha='center', xycoords='axes fraction',color='k',fontsize=12);
# Dry
ax0 = fig.add_subplot(spec[2:4,2:4], projection=plot_proj,transform=plot_proj)
ax0.set_title('D',fontsize=14,fontweight='bold',loc='left');
ax0.coastlines(color='grey'); 
subplot_num=0
patch=[]
x = dry_regions_combined
for region in list(x.keys()):
    xnonan = x.fillna(0)
    xnonan[region].plot.contour(ax=ax0, levels=np.linspace(0,1.1,2), colors=region_colors[subplot_num], linewidth=0.5, transform= ccrs.PlateCarree(), add_labels=region);
    patch.append(mpatches.Patch(color=region_colors[subplot_num], label=region))
    subplot_num+=1
plt.legend(loc='upper center',handles=patch,ncol=2,bbox_to_anchor=(0.5,0));
plt.annotate('Dry', xy=(0.5,1.05), ha='center', xycoords='axes fraction',color='k',fontsize=12);
# Figure aesthetics
plt.tight_layout();





#%% CALCULATE POP WEIGHTED T AND P ANOMALIES (for Table S2)
## Read in pop data
# Download raw data from: 
# Center For International Earth Science Information Network-CIESIN-Columbia University. (2017). 
# Gridded Population of the World, Version 4 (GPWv4): Population Density, Revision 11
# (Version 4.11) [Data set]. Palisades, NY: NASA Socioeconomic Data and Applications Center (SEDAC).
#  https://doi.org/10.7927/H49C6VHW 
## 2010 ##
# This is just a placeholder. You must download and regrid the data to the target grid.
wd = '/_data/pop_data/gpw-v4-population-count-rev11_totpop_30_min_nc/'
regrid_pop_count = xr.open_dataset(wd+'gpw-v4-population-count-rev11_totpop_192x288.nc')
pop_count = regrid_pop_count.sel(time=2015).pop_count
# Turn ocean values to nan
landmask = atm_monthly_ctrl['']['LANDFRAC'].isel(time=0,member=0).load()
pop_count_subset = xr.where((landmask>0.1), pop_count, np.nan)
# Compute GDP weights
pop_wt = pop_count_subset/np.nansum(pop_count_subset)

## Weight absolute T and P anomalies by global population and compute global sum of weighted anomalies for each MCB case
# MCB
ts_mcb_abs_anom_pop_wt = {}
prect_mcb_abs_anom_pop_wt = {}
for key in mcb_keys:
    ts_mcb_abs_anom_pop_wt[key] = np.nansum(np.abs(ts_mcb_anom_jun_aug[key])*pop_wt)
    prect_mcb_abs_anom_pop_wt[key] = np.nansum(np.abs(prect_mcb_anom_jun_aug[key])*pop_wt)



