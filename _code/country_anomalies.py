### PURPOSE: Script to calculate E/C indices from surface temperature EOFs
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 08/27/2024
### Note: script adapted from smyle_mcb_eof_v3.py
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
import lens2_preanalysis_functions as fun
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

plt.ion();

dask.config.set({"array.slicing.split_large_chunks": False})

#run this line in console before starting ipython if getting OMP_NUM_THREADS error
os.environ["OMP_NUM_THREADS"] = "1"

##################################################################################################################
## WHICH EXPERIMENT ARE YOU READING IN? ##
# month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
year_init = input('Which initialization year are you reading in (1997, 2015?): ')
# enso_phase = input('Which ENSO event are you reading in (nino or nina)?: ')
# sensitivity_opt = input('Sensitivity run (y or n)?: ')
# Hard code for 2015 testing
month_init = '05'
# year_init = '2015'
enso_phase = 'nino'
sensitivity_opt = 'y'
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
    ctrl_files = ctrl_files + glob.glob('/_data/SMYLE-MCB/realtime/b.e21.BSMYLE.f09_g17.'+yr+'*-'+month_init+'.*')
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
            mcb_files = mcb_files + glob.glob('/_data/SMYLE-MCB/MCB/b.e21.BSMYLE.f09_g17.MCB*'+yr+'*-'+month_init+'_'+key+'.*')
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
            mcb_files = mcb_files + glob.glob('/_data/SMYLE-MCB/MCB/b.e21.BSMYLE.f09_g17.MCB.'+yr+'*-'+month_init+'.*')
        mcb_members = []
        for i in mcb_files:
            start = i.find('f09_g17.MCB') + len('f09_g17.MCB.')
            tmp = i[start:None]
            if tmp not in mcb_members:
                mcb_members.append(tmp)
        mcb_members = sorted(mcb_members)
        print(mcb_members)
        mcb_sims[key] = mcb_members 


# Get list of control climatology ensemble members
clim_files =  glob.glob('/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/TS/b.e21.BSMYLE.f09_g17.1970-'+month_init+'*.nc')
clim_members = []
for i in clim_files:
    start = i.find('f09_g17.1970-'+month_init+'.') + len('f09_g17.1970-'+month_init+'.')
    tmp = i[start:start+3]
    if tmp not in clim_members:
        clim_members.append(tmp)
clim_members = sorted(clim_members)
print(clim_members) 


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
# Do you need to make processed climatology file?
# process_opt = input('Do you need to make processed historical file? (y or n): ')
process_opt = 'n'
# Set target directory for processed files
target_dir='/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed'
# Make the target directory if necessary
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
# Process each ensemble member and save as a concatenated file with ensemble member as a dimension
if process_opt=='y':
    atm_monthly_ctrl_clim = {}
    for m in clim_members:
        print(m)
        combined_vars=xr.Dataset()
        for var in atm_varnames_monthly_subset:
            file_subset_clim =  sorted(glob.glob('/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/'+var+'/b.e21.BSMYLE.f09_g17.*'+m+'.cam*'))
            for file in file_subset_clim:
                if file_subset_clim.index(file)==0:
                    da_merged = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var]
                else:
                    next_file = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var]
                    da_merged = xr.concat([da_merged, next_file], dim='time')
            combined_vars=xr.merge([combined_vars,da_merged])
        atm_monthly_ctrl_clim[m] = combined_vars
    # Combine all files into one xarray dataset with ensemble members as a new dimension
    atm_monthly_ctrl_clim_xr = xr.concat(list(map(atm_monthly_ctrl_clim.get, clim_members)),pd.Index(clim_members,name='member'))
    ## Convert time to datetime index
    atm_monthly_ctrl_clim_xr = atm_monthly_ctrl_clim_xr.assign_coords(time=atm_monthly_ctrl_clim_xr.indexes['time'].to_datetimeindex())
    ## Convert units
    # PRECT
    m_to_mm = 1e3 #mm/m
    s_to_days = 86400 #s/day
    # Convert from m/s to mm/day
    atm_monthly_ctrl_clim_xr = atm_monthly_ctrl_clim_xr.assign(PRECT=atm_monthly_ctrl_clim_xr['PRECT']*m_to_mm*s_to_days)
    atm_monthly_ctrl_clim_xr['PRECT'].attrs['units'] = 'mm/day'
    # TS
    # Convert from K to C
    atm_monthly_ctrl_clim_xr = atm_monthly_ctrl_clim_xr.assign(TS=atm_monthly_ctrl_clim_xr['TS']-273.15)
    atm_monthly_ctrl_clim_xr['TS'].attrs['units'] = '°C'
    ### EXPORT PROCESSED NETCDF
    atm_monthly_ctrl_clim_xr.to_netcdf(target_dir+'/BSMYLE.'+str(pd.to_datetime(atm_monthly_ctrl_clim_xr.time.values[0]).year)+'-'+str(pd.to_datetime(atm_monthly_ctrl_clim_xr.time.values[-1]).year)+'-'+month_init+'.TS_PRECT_concat.nc',mode='w',format='NETCDF4')
elif process_opt=='n':
    # Skip to read in pre-processed file
    atm_monthly_ctrl_clim_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*TS_PRECT_concat.nc')[0])))


## COMPUTE LONG TERM STANDARD DEVIATION AND MONTHLY CLIMATOLOGY MEAN FROM 1970-2014
# Subset time from 1970-2014
hist_ext = atm_monthly_ctrl_clim_xr.isel(time=atm_monthly_ctrl_clim_xr['time.year']<2015)
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
        dir_ctrl = '/_data/SMYLE-MCB/realtime/b.e21.BSMYLE.f09_g17.'+m+'/atm/proc/tseries/month_1'
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
    ##DRIFT CORRECTION
    # Compute drift correction anomaly
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_ctrl_copy = atm_monthly_ctrl[key]['TS']*1
    prect_ctrl_copy = atm_monthly_ctrl[key]['PRECT']*1
    # Compute climatological anomalies
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
        dir_mcb = glob.glob('/_data/SMYLE-MCB/MCB/b.e21.BSMYLE.f09_g17.MCB*'+m+'/atm/proc/tseries/month_1')[0]
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
    ##DRIFT CORRECTION
    # Compute drift correction anomaly
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_mcb_copy = atm_monthly_mcb[key]['TS']*1
    prect_mcb_copy = atm_monthly_mcb[key]['PRECT']*1 
    # Compute climatological anomalies
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
seeding_mask = fun.reorient_netCDF(xr.open_dataset('/home/j4wan/SMYLE-MCB/processed_data/mask_CESM/sesp_mask_CESM2_0.9x1.25_v3.nc'))

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



#%% E/C-INDEX CALCULATIONS
# # Adapted from Callahan & Mankin (2023) Observed_ENSO_Indices.ipynb and CMIP6_ENSO_Indices
## Define function for detrending data
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
    # timevals = xr.DataArray(np.arange(1,len(tm)+1,1),
    #                     coords=[tm],
    #                     dims=["time"])
    timevals = data['time.year']+(data.time.dt.dayofyear/365)
    timevals = timevals.expand_dims(lat=lt,lon=ln)
    timevals = timevals.transpose("time","lat","lon")
    
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



#%% CALCULATE POP WEIGHTED T AND P ANOMALIES
## Read in pop data
# Read in regridded population data (as used in Wan et al., 2024)
## 2010 ##
wd = '/home/j4wan/NCAR-ECIP/pop_data/gpw-v4-population-count-rev11_totpop_30_min_nc/'
regrid_pop_count = xr.open_dataset(wd+'gpw-v4-population-count-rev11_totpop_192x288.nc')
pop_count = regrid_pop_count.sel(time=2015).pop_count
# Turn ocean values to nan
landmask = atm_monthly_ctrl['']['LANDFRAC'].isel(time=0,member=0).load()
pop_count_subset = xr.where((landmask>0.1), pop_count, np.nan)
# Compute GDP weights
pop_wt = pop_count_subset/np.nansum(pop_count_subset)


# Read in LENS2 SDs
data_dir = '/home/j4wan/SMYLE-MCB/processed_data/LENS2/'
lens2_ts_sd = fun.reorient_netCDF(xr.open_dataset(data_dir+'CESM-LENS2.hist.ensemble.std.TS.1970-2014.nc')).TS
lens2_prect_sd = fun.reorient_netCDF(xr.open_dataset(data_dir+'CESM-LENS2.hist.ensemble.std.PRECT.1970-2014.nc')).PRECT
# Align grid with CESM grid (fix rounding errors from regridding)
lens2_ts_sd = lens2_ts_sd.assign_coords({'lon':ts_ctrl_anom[''].lon,\
                            'lat':ts_ctrl_anom[''].lat})
lens2_prect_sd = lens2_prect_sd.assign_coords({'lon':ts_ctrl_anom[''].lon,\
                            'lat':ts_ctrl_anom[''].lat})


### DETREND ANOMALIES AND LANDMASKED SD NORMALIZED ANOMALIES
## Load in all anomaly files at once (required for detrending) and convert to standard deviation space
# CONTROL
ts_ctrl_anom_load = {}
prect_ctrl_anom_load = {}
for key in ctrl_keys:
    ts_ctrl_anom_load[key] = ts_ctrl_anom[key].load()/lens2_ts_sd
    prect_ctrl_anom_load[key] = prect_ctrl_anom[key].load()/lens2_prect_sd

# MCB
ts_mcb_anom_load = {}
prect_mcb_anom_load = {}
for key in mcb_keys:
    print(key)
    ts_mcb_anom_load[key] = ts_mcb_anom[key].load()/lens2_ts_sd
    prect_mcb_anom_load[key] = prect_mcb_anom[key].load()/lens2_prect_sd

# HISTORICAL
ts_hist_anom_load = ts_hist_anom.load()/lens2_ts_sd
prect_hist_anom_load = prect_hist_anom.load()/lens2_prect_sd


## Detrend with the historical predicted values
# HISTORICAL
ts_hist_detrend, ts_hist_detrend_resid = xarray_linear_detrend(ts_hist_anom_load.mean(dim='member'))
prect_hist_detrend, prect_hist_detrend_resid = xarray_linear_detrend(prect_hist_anom_load.mean(dim='member'))
# Mask out land
ts_hist_detrend_sd_land= xr.where((landmask>0.1),ts_hist_detrend,np.nan)
prect_hist_detrend_sd_land= xr.where((landmask>0.1),prect_hist_detrend,np.nan)


# Create continuous time series by averaging over overlapping periods for El Niño years
tcont = np.unique(ts_ctrl_anom_load[''].time.values)
ts_hist_detrend_resid_subset = ts_ctrl_anom_load[''].isel(member=0)*np.nan
prect_hist_detrend_resid_subset = prect_ctrl_anom_load[''].isel(member=0)*np.nan
for t in tcont:
    t_unique = ts_hist_detrend_resid.where((ts_hist_detrend_resid.time==t),drop=True).mean(dim='time')
    ts_hist_detrend_resid_subset = xr.where((ts_hist_detrend_resid_subset.time==t),t_unique,ts_hist_detrend_resid_subset)
    prect_hist_detrend_resid_subset = xr.where((prect_hist_detrend_resid_subset.time==t),t_unique,prect_hist_detrend_resid_subset)


# CONTROL
ts_ctrl_detrend_land_sd = {}
prect_ctrl_detrend_land_sd = {}
for key in ctrl_keys:
    # Mask out land
    ts_sd_land_only = xr.where((landmask>0.1),ts_ctrl_anom_load[key],np.nan)
    prect_sd_land_only = xr.where((landmask>0.1),prect_ctrl_anom_load[key],np.nan)
    # Detrend with historical values
    ts_ctrl_detrend_land_sd[key] = ts_sd_land_only.mean(dim='member')-ts_hist_detrend_resid_subset
    prect_ctrl_detrend_land_sd[key] = prect_sd_land_only.mean(dim='member')-prect_hist_detrend_resid_subset

## MCB
ts_mcb_detrend_land_sd = {}
prect_mcb_detrend_land_sd = {}
for key in mcb_keys:
    print(key)
    # Mask out land
    ts_sd_land_only = xr.where((landmask>0.1),ts_mcb_anom_load[key],np.nan)
    prect_sd_land_only = xr.where((landmask>0.1),prect_mcb_anom_load[key],np.nan)
    # Detrend with historical values
    ts_mcb_detrend_land_sd[key] = ts_sd_land_only.mean(dim='member')-ts_hist_detrend_resid_subset
    prect_mcb_detrend_land_sd[key] = prect_sd_land_only.mean(dim='member')-prect_hist_detrend_resid_subset


### PLOT MAPS OF NORMALIZED ANOMALIES (w/ POP WEIGHTING)
# Create El Niño DJF average
if year_init=='1997':
    peak_yrs = [1997,1998]
elif year_init=='2015':
    peak_yrs = [2015,2016]
# djf_index = ts_ctrl_eindex[''].loc[{'time':[t for t in pd.to_datetime(ts_ctrl_eindex[''].time.values) if (t.year==peak_yrs[0])&(t.month==12)|\
#                                                         (t.year==peak_yrs[0]+1)&(t.month==1)|\
#                                                         (t.year==peak_yrs[0]+1)&(t.month==2)]}].get_index('time')
# # Create June-June subset
# june_index = ts_ctrl_eindex[''].loc[{'time':[t for t in pd.to_datetime(ts_ctrl_eindex[''].time.values) if (t.year==peak_yrs[0])&(t.month>=6)|\
#                                                         (t.year==peak_yrs[0]+1)&(t.month<=6)]}].get_index('time')
def djf_peak_subset(data):
    djf_peak = data.where(((data['time.year']==peak_yrs[0])&(data['time.month']==12))|\
                                        ((data['time.year']==peak_yrs[1])&(data['time.month']==1))|\
                                        ((data['time.year']==peak_yrs[1])&(data['time.month']==2)),drop=True)
    return djf_peak

# Create June-June subset
def june_subset(data):
    june_subset = data.where(((data['time.year']==peak_yrs[0])&(data['time.month']>=6))|\
                                        ((data['time.year']==peak_yrs[0])&(data['time.month']<=6))\
                                        ,drop=True)
    return june_subset

# Define function to calculate DJF means for each year with discontinuous time
def djf_mean_annual(data):
    # data: monthly dataarray over which you want to calculate DJF means
    djf_subset = data.where((data.month==1)|(data.month==2)|(data.month==12),drop=True)
    djf_rolling = djf_subset.rolling(time=3,center=True).mean()
    djf_xr = djf_rolling.isel(time=djf_rolling['time.month']==1)
    return djf_xr



#%% METHOD 2: CALCULATE COUNTRY LEVEL POPULATION WEIGHTED T AND P ANOMALIES
# Read in country geometry file
countries = gpd.read_file('/home/j4wan/Migration/projections/country_shp/ne_50m_admin_0_countries.shp')
countries = countries.rename(columns={'ISO_N3':'country_id'})
# Norway (ISO_N3=-99, need to manually add to dataframe)
countries.loc[88,'ISO_A3']='NOR'
countries.loc[88,'country_id']='578'
# Remove -99 country_id (5 rows)
countries = countries.drop(countries[countries['country_id']=='-99'].index)
countries['CountryID'] = countries.country_id.astype(int).astype(str).str.zfill(3).astype(float)

# Read in country pop data
pop_country_df = pd.read_csv('/home/j4wan/SMYLE-MCB/processed_data/gpw/gpw-v4-national-identifier-popcount-2015.csv', index_col=[0])

# Subset peak DJF s.d. normalized anomalies
ts_ctrl_detrend_peak = {}
ts_mcb_detrend_peak = {}

prect_ctrl_detrend_peak = {}
prect_mcb_detrend_peak = {}

# Compute for DJF annual means
# HISTORICAL
ts_hist_detrend_peak = djf_mean_annual(ts_hist_detrend_sd_land).groupby('time.year').mean()
prect_hist_detrend_peak = djf_mean_annual(prect_hist_detrend_sd_land).groupby('time.year').mean()

# CONTROL
for key in ctrl_keys:
    ts_ctrl_detrend_peak[key] = djf_mean_annual(ts_ctrl_detrend_land_sd[key]).groupby('time.year').mean()
    prect_ctrl_detrend_peak[key] = djf_mean_annual(prect_ctrl_detrend_land_sd[key]).groupby('time.year').mean()

# MCB
for key in mcb_keys:
    print(key)
    ts_mcb_detrend_peak[key] = djf_mean_annual(ts_mcb_detrend_land_sd[key]).groupby('time.year').mean()
    prect_mcb_detrend_peak[key] = djf_mean_annual(prect_mcb_detrend_land_sd[key]).groupby('time.year').mean()


# Convert to geopandas dataframe to calculate country level population weighted anomalies
combined_detrend_peak_country_gc = pd.DataFrame()
## HISTORICAL
# Convert anomalies xarray into dataframe
ts_gc_df = ts_hist_detrend_peak.to_dataframe(name='TS_anom').reset_index()
prect_gc_df =  prect_hist_detrend_peak.to_dataframe(name='PRECT_anom').reset_index()
anom_gc_df = pd.merge(ts_gc_df, prect_gc_df,on=('year', 'lat','lon'))
# Add ensemble member as a column
anom_gc_df['exp_id'] = 'Historical' 
# Merge anomalies and population data into one dataframe
tmp_gc_countries_df = anom_gc_df.merge(pop_country_df,on=('lat','lon'),how='left')
combined_detrend_peak_country_gc = pd.concat([combined_detrend_peak_country_gc,tmp_gc_countries_df], ignore_index=True)
## CONTROL 
for key in ctrl_keys:
    print(key)
    # Convert anomalies xarray into dataframe
    ts_gc_df = ts_ctrl_detrend_peak[key].to_dataframe(name='TS_anom').reset_index()
    prect_gc_df =  prect_ctrl_detrend_peak[key].to_dataframe(name='PRECT_anom').reset_index()
    anom_gc_df = pd.merge(ts_gc_df, prect_gc_df,on=('year', 'lat','lon'))
    # Add ensemble member as a column
    anom_gc_df['exp_id'] = 'Control' 
    # Merge anomalies and population data into one dataframe
    tmp_gc_countries_df = anom_gc_df.merge(pop_country_df,on=('lat','lon'),how='left')
    combined_detrend_peak_country_gc = pd.concat([combined_detrend_peak_country_gc,tmp_gc_countries_df], ignore_index=True)
## MCB 
for key in mcb_keys:
    print(key)
    # Convert anomalies xarray into dataframe
    ts_gc_df = ts_mcb_detrend_peak[key].to_dataframe(name='TS_anom').reset_index()
    prect_gc_df =  prect_mcb_detrend_peak[key].to_dataframe(name='PRECT_anom').reset_index()
    anom_gc_df = pd.merge(ts_gc_df, prect_gc_df,on=('year', 'lat','lon'))
    # Add ensemble member as a column
    anom_gc_df['exp_id'] = key 
    # Merge anomalies and population data into one dataframe
    tmp_gc_countries_df = anom_gc_df.merge(pop_country_df,on=('lat','lon'),how='left')
    combined_detrend_peak_country_gc = pd.concat([combined_detrend_peak_country_gc,tmp_gc_countries_df], ignore_index=True)

# Remove rows with NaN CountryIDs
combined_detrend_peak_country_gc = combined_detrend_peak_country_gc[combined_detrend_peak_country_gc.ISOCODE.notnull()]
# Remove time column from pop data 
combined_detrend_peak_country_gc = combined_detrend_peak_country_gc.drop(columns='time')
# Merge anomalies and population with geopandas geometry 
combined_detrend_peak_country_gc_geo = gpd.GeoDataFrame(combined_detrend_peak_country_gc, \
                                    geometry=gpd.points_from_xy(combined_detrend_peak_country_gc.lon, combined_detrend_peak_country_gc.lat))
# Add column for sum of abosulte values of T and P
combined_detrend_peak_country_gc_geo['Sum_abs_anom'] = abs(combined_detrend_peak_country_gc_geo['TS_anom'])+abs(combined_detrend_peak_country_gc_geo['PRECT_anom'])
# Calculate country population sum
country_sums = combined_detrend_peak_country_gc.groupby(['year','exp_id','CountryID'])[['pop_count']].sum().reset_index().rename(columns={'pop_count':'country_sum_popc'})
combined_detrend_peak_country_gc_geo = combined_detrend_peak_country_gc_geo.merge(country_sums,on=('year','CountryID','exp_id'),how='left')
# Calculate country level population weight
combined_detrend_peak_country_gc_geo['country_popw'] = combined_detrend_peak_country_gc_geo['pop_count']/combined_detrend_peak_country_gc_geo['country_sum_popc'] 
# Weight anomalies by population weight
combined_detrend_peak_country_gc_geo['TS_anom_popw'] =  combined_detrend_peak_country_gc_geo['TS_anom']*combined_detrend_peak_country_gc_geo['country_popw']
combined_detrend_peak_country_gc_geo['PRECT_anom_popw'] =  combined_detrend_peak_country_gc_geo['PRECT_anom']*combined_detrend_peak_country_gc_geo['country_popw']
combined_detrend_peak_country_gc_geo['Sum_abs_anom_popw'] =  combined_detrend_peak_country_gc_geo['Sum_abs_anom']*combined_detrend_peak_country_gc_geo['country_popw']

# Aggregate to country-level means
combined_detrend_peak_country_level_geo_means = combined_detrend_peak_country_gc_geo.groupby(['year','CountryID','exp_id'])\
                                                ['TS_anom_popw','TS_anom','PRECT_anom_popw','PRECT_anom','Sum_abs_anom_popw','Sum_abs_anom'].sum().reset_index()
# Add back in columns for plotting
combined_detrend_peak_country_level_geo_means = pd.merge(combined_detrend_peak_country_level_geo_means,countries[['ISO_A3','CountryID','geometry']],on='CountryID',how='left')
# Remove any NaN rows introduced with ISO_A3
combined_detrend_peak_country_level_geo_means = combined_detrend_peak_country_level_geo_means[combined_detrend_peak_country_level_geo_means.ISO_A3.notnull()]
# Replace -99 ISO_A3 based on CountryID values (36==AUS; 250==FRA)
combined_detrend_peak_country_level_geo_means.loc[combined_detrend_peak_country_level_geo_means['CountryID']==36.0,'ISO_A3']='AUS'
combined_detrend_peak_country_level_geo_means.loc[combined_detrend_peak_country_level_geo_means['CountryID']==250.0,'ISO_A3']='FRA'

# Export country-level dataframe as csv
combined_detrend_peak_country_level_geo_means.to_csv('_data/SMYLE-MCB/processed_data/callahan_regression/country_popw_anom_djf_'+year_init+'-'+month_init+'_v2.csv')
combined_detrend_peak_country_level_geo_means.drop(columns='geometry').to_csv('/_data/SMYLE-MCB/processed_data/callahan_regression/country_popw_anom_djf_nogeo_'+year_init+'-'+month_init+'_v2.csv')

## Read in E and C index from smyle_fosi_eindex.py
e_c_index_df = pd.read_csv('/home/j4wan/SMYLE-MCB/processed_data/callahan_regression/e-c_index_djf_timeseries_'+year_init+'-'+month_init+'_v6.csv', index_col=[0])
e_c_index_df = e_c_index_df.rename(columns={'time':'year','experiment':'exp_id'})


# Plot scatter of combined anomalies and e-index
# Set marker colors and styles
mcb_colors = {'Control':'k','06-02':'#a50f15','06-08':'#a50f15','06-11':'#a50f15','09-02':'#ef3b2c','09-11':'#ef3b2c','12-02':'#fc9272'} # reds=start month
mcb_marker =  {'Control':'o','06-02':'*','06-08':'^','06-11':'P','09-02':'P','09-11':'^','12-02':'^'} # duration
# Create plot
subplot_num=0
plt.figure(figsize=(8,6));
plt.subplot(1,1,subplot_num+1);
# Historical DJF means (Control)
plt.scatter(e_c_index_df[e_c_index_df['exp_id']=='Control'][1:]['E-index'],\
            combined_detrend_peak_country_level_geo_means[combined_detrend_peak_country_level_geo_means['exp_id']=='Historical'].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c='grey',alpha=0.25,label='Historical (Control)');
# Historical DJF means (MCB)
plt.scatter(e_c_index_df[(e_c_index_df['MCB_year']=='N')].groupby('year').mean()[1:]['E-index'],\
            combined_detrend_peak_country_level_geo_means[combined_detrend_peak_country_level_geo_means['exp_id']=='Historical'].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c='b',alpha=0.25,label='Historical (MCB mean)');
# Subset of El Niño event
plt.scatter(e_c_index_df[(e_c_index_df['exp_id']=='Control')&(e_c_index_df['year']==peak_yrs[1])]['E-index'],\
            combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']=='Control')&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c='k',alpha=1, s=100, label='Control');
### MCB
# Subset of El Niño event
for key in mcb_keys:
    plt.scatter(e_c_index_df[(e_c_index_df['exp_id']==key)&(e_c_index_df['year']==peak_yrs[1])&(e_c_index_df['MCB_year']=='Y')]['E-index'],\
            combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']==key)&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c=mcb_colors[key],marker=mcb_marker[key], alpha=1, s=100, label=key);
subplot_num+=1
if subplot_num==1:
    plt.legend(loc='lower right');
plt.grid();
plt.axhline(0,color='k',linewidth=2);
plt.axvline(0,color='k',linewidth=2);
plt.xlabel('E-index (s.d.)', fontsize=12);
plt.ylabel('|T|+|P| (s.d.)', fontsize=12);
plt.ylim(-0.1,1);plt.xlim(-2,5);
plt.title(year_init+' El Niño (DJF)',fontsize=12,fontweight='bold',loc='left');



#%% COMPUTE COUNTRY-LEVEL CORRECTION FACTORS
# Get list of unique countries
country_list = list(combined_detrend_peak_country_level_geo_means.ISO_A3.unique())
# # Remove -99 country code
# country_list.remove('-99')
# Sample country
c = combined_detrend_peak_country_level_geo_means.ISO_A3.unique()[0]

country_correction_df = pd.DataFrame()
for c in country_list:
    ## Isolate El Niño peak DJF x,y coordinate of MCB and control (E index and Abs anom)
    ycoord = combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['ISO_A3']==c)&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])][['exp_id','ISO_A3','year','Sum_abs_anom_popw']]
    xcoord_mcb = e_c_index_df[(e_c_index_df['year']==peak_yrs[1])&(e_c_index_df['MCB_year']=='Y')]
    xcoord_control = e_c_index_df[(e_c_index_df['year']==peak_yrs[1])&(e_c_index_df['exp_id']=='Control')]
    xcoord = pd.concat([xcoord_mcb,xcoord_control])
    xycoords = pd.merge(ycoord,xcoord[['year','E-index','exp_id']],on=('exp_id','year'),how='right').drop(columns='year')
    ## Calculate y-intercept relative to control
    blist = []
    for key in xycoords.exp_id:
        if key!='Control':
            b = np.polyfit([float(np.unique(xycoords[xycoords['exp_id']==key]['E-index'].values)),float(np.unique(xycoords[xycoords['exp_id']=='Control']['E-index'].values))],\
                                    [float(np.unique(xycoords[xycoords['exp_id']==key]['Sum_abs_anom_popw'].values)),float(np.unique(xycoords[xycoords['exp_id']=='Control']['Sum_abs_anom_popw'].values))],1)[1]
            # y-int is calculated relative to control y
            blist.append(float(np.unique(xycoords[xycoords['exp_id']=='Control']['Sum_abs_anom_popw'].values))-b)
        # No intercept for control (same point)
        elif key=='Control':
            blist.append(np.nan)
    xycoords['b'] =  blist
    ## Calculate historical y-intercept
    # Add absolute value of E-index as a column
    e_c_index_df = e_c_index_df.assign(Abs_eindex = np.abs(e_c_index_df['E-index']))
    # Find 10 lowest E-index years from historical points
    neutral_hist_yrs = e_c_index_df[e_c_index_df['exp_id']=='Control'].sort_values('Abs_eindex')[:10]['year']
    # Compute mean E-index for the neutral historical years
    xcoord_hist = e_c_index_df[e_c_index_df['exp_id']=='Control'].sort_values('Abs_eindex')[:10]['E-index'].mean()
    # Compute mean country-level anomalies for the neutral historical years
    ycoord_hist = combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['ISO_A3']==c)][['exp_id','ISO_A3','year','Sum_abs_anom_popw']]
    ycoord_hist = ycoord_hist[ycoord_hist['year'].isin(neutral_hist_yrs)]['Sum_abs_anom_popw'].mean()
    # Calculate historical intercept
    b_hist = float(np.unique(xycoords[xycoords['exp_id']=='Control']['Sum_abs_anom_popw'].values))-np.polyfit([xcoord_hist,float(np.unique(xycoords[xycoords['exp_id']=='Control']['E-index'].values))],\
                        [ycoord_hist,float(np.unique(xycoords[xycoords['exp_id']=='Control']['Sum_abs_anom_popw'].values))],1)[1]
    # Assign historical values to xycoords df
    df_hist = pd.DataFrame([['Historical',c,ycoord_hist,xcoord_hist,b_hist]],columns=['exp_id','ISO_A3','Sum_abs_anom_popw','E-index','b'])
    xycoords = pd.concat([xycoords,df_hist],ignore_index=True)
    # Calculate correction factor (alpha)
    xycoords = xycoords.assign(alpha = lambda x: x.b/(x[x['exp_id']=='Historical'].b.values))
    # # Remove Control and Historical rows (only want MCB correction factors)
    # xycoords = xycoords[(xycoords['exp_id']!='Control')&(xycoords['exp_id']!='Historical')]
    # Append xycoords for individual country to full country df
    country_correction_df = pd.concat([country_correction_df,xycoords],ignore_index=True)

# Export country-level correction factors dataframe as csv
country_correction_df.to_csv('/_data/SMYLE-MCB/processed_data/callahan_regression/country_correction_factor_djf_peak_'+year_init+'-'+month_init+'_v1.csv')


# Plot multi-panel scatter of absolute anomalies and e-index
version_num=1
# Set marker colors and styles
mcb_colors = {'Control':'k','06-02':'#a50f15','06-08':'#a50f15','06-11':'#a50f15','09-02':'#ef3b2c','09-11':'#ef3b2c','12-02':'#fc9272'} # reds=start month
mcb_marker =  {'Control':'o','06-02':'*','06-08':'^','06-11':'P','09-02':'P','09-11':'^','12-02':'^'} # duration
mcb_legend_label = {'06-02':'Jun-Feb','06-08':'Jun-Aug','06-11':'Jun-Nov','09-02':'Sep-Feb','09-11':'Sep-Nov','12-02':'Dec-Feb'}
subplot_label = ['a','b','c','d']
# Create plot
subplot_num=0
plt.figure(figsize=(8,8));
## a) GLOBAL
plt.subplot(2,2,subplot_num+1);
# Historical DJF means (Control)
plt.scatter(e_c_index_df[e_c_index_df['exp_id']=='Control'][1:]['E-index'],\
            combined_detrend_peak_country_level_geo_means[combined_detrend_peak_country_level_geo_means['exp_id']=='Historical'].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c='grey',alpha=0.25,label='Historical');
# Subset of El Niño event
plt.scatter(e_c_index_df[(e_c_index_df['exp_id']=='Control')&(e_c_index_df['year']==peak_yrs[1])]['E-index'],\
            combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']=='Control')&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c='k',marker='s',alpha=1, s=100, label='Control');
### MCB
# Subset of El Niño event
for key in mcb_keys:
    plt.scatter(e_c_index_df[(e_c_index_df['exp_id']==key)&(e_c_index_df['year']==peak_yrs[1])&(e_c_index_df['MCB_year']=='Y')]['E-index'],\
            combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']==key)&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])].groupby('year').mean()['Sum_abs_anom_popw']\
            ,c=mcb_colors[key],marker=mcb_marker[key], alpha=1, s=100, label=mcb_legend_label[key]);
plt.ylim(-0.1,1);plt.xlim(-2,5);
if subplot_num==0:
    if year_init=='2015':
        plt.legend(loc='lower right')
    elif year_init=='1997':
        plt.legend(loc='upper left');
plt.grid();
plt.axhline(0,color='k',linewidth=2);
plt.axvline(0,color='k',linewidth=2);
# plt.xlabel('E-index (s.d.)', fontsize=12);
plt.ylabel('|T|+|P| (s.d.)', fontsize=12);
plt.annotate('Global',xy=(.95,.95),xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=12);
plt.title(subplot_label[subplot_num],fontsize=12,fontweight='bold',loc='left');
subplot_num+=1

## b-d) Correction annotations 
country_subset = ['ECU','IDN','USA']
for c_plot in country_subset:
    plt.subplot(2,2,subplot_num+1);
    # Historical DJF means (Control)
    plt.scatter(e_c_index_df[e_c_index_df['exp_id']=='Control'][1:]['E-index'],\
                combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']=='Historical')\
                &(combined_detrend_peak_country_level_geo_means['ISO_A3']==c_plot)].groupby('year').mean()['Sum_abs_anom_popw']\
                ,c='grey',alpha=0.25,label='Historical');
    # Subset of El Niño event
    plt.scatter(e_c_index_df[(e_c_index_df['exp_id']=='Control')&(e_c_index_df['year']==peak_yrs[1])]['E-index'],\
                combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']=='Control')&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])\
                &(combined_detrend_peak_country_level_geo_means['ISO_A3']==c_plot)].groupby('year').mean()['Sum_abs_anom_popw']\
                ,c='k',marker='s',alpha=1, s=100, label='Control');
    ### MCB
    # Subset of El Niño event
    for key in mcb_keys:
        plt.scatter(e_c_index_df[(e_c_index_df['exp_id']==key)&(e_c_index_df['year']==peak_yrs[1])&(e_c_index_df['MCB_year']=='Y')]['E-index'],\
                combined_detrend_peak_country_level_geo_means[(combined_detrend_peak_country_level_geo_means['exp_id']==key)&(combined_detrend_peak_country_level_geo_means['year']==peak_yrs[1])\
                &(combined_detrend_peak_country_level_geo_means['ISO_A3']==c_plot)].groupby('year').mean()['Sum_abs_anom_popw']\
                ,c=mcb_colors[key],marker=mcb_marker[key], alpha=1, s=100, label=mcb_legend_label[key]);
    ## Add country correction annotations
    # Horizontal line for control y
    exp_id = 'Control'
    control_x = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['E-index'].values);
    control_y = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['Sum_abs_anom_popw'].values);
    plt.axhline(control_y,linestyle='dotted',color='k');
    # Linear fit from control to Full effort
    exp_id = '06-02'
    mcb_x = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['E-index'].values);
    mcb_y = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['Sum_abs_anom_popw'].values);
    mcb_b = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['b'].values);
    plt.plot([mcb_x,control_x],[mcb_y,control_y],linestyle='dashed',color=mcb_colors[exp_id]);
    # b_mcb annotation
    plt.plot([control_x*1.4,control_x*1.4], [control_y-mcb_b, control_y], marker='_',color=mcb_colors[exp_id]);
    plt.annotate('$b_{MCB}$', xy=(control_x*1.5,mcb_y+(control_y-mcb_y)/2),color=mcb_colors[exp_id],fontsize=10);
    # Linear fit from control to Historical mean
    exp_id = 'Historical'
    hist_x = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['E-index'].values);
    hist_y = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['Sum_abs_anom_popw'].values);
    hist_b = float(country_correction_df[(country_correction_df['exp_id']==exp_id)&(country_correction_df['ISO_A3']==c_plot)]['b'].values);
    plt.plot([hist_x,control_x],[hist_y,control_y],linestyle='dashed',color='k');
    # b_historical annotation
    plt.plot([control_x*1.2,control_x*1.2], [control_y-hist_b, control_y], marker='_',color='k');
    plt.annotate('$b_{historical}$', xy=(control_x*1.3,hist_y+(control_y-hist_y)/2),color='k',fontsize=10);
    if subplot_num==0:
        plt.legend(loc='lower right');
    plt.grid();
    plt.axhline(0,color='k',linewidth=2);
    plt.axvline(0,color='k',linewidth=2);
    if (subplot_num==2)|(subplot_num==3):
        plt.xlabel('E-index (s.d.)', fontsize=12);
    if (subplot_num==2):
        plt.ylabel('|T|+|P| (s.d.)', fontsize=12);
    plt.annotate(c_plot,xy=(.95,.95),xycoords='axes fraction',horizontalalignment='right', verticalalignment='top',fontsize=12);
    plt.annotate(r'$\alpha$ $= b_{MCB}/b_{historical}$',xy=(1,0.05),xycoords='axes fraction',horizontalalignment='right', verticalalignment='bottom',fontsize=10);
    plt.title(subplot_label[subplot_num],fontsize=12,fontweight='bold',loc='left');
    subplot_num+=1
plt.tight_layout();


#%% TOTAL SHIP OPERATION COST ESTIMATE
month_day_dict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
n_ships = 2400 #estimated from Wood 2021
daily_ship_op_cost = 7474 #taken from Ship Operating Costs Annual Review and Forecast 2022/23 Drewry report

total_ship_op_cost = {}
for key in mcb_keys:
    if key=='06-02':
        mcb_months = [1,2,6,7,8,9,10,11,12]
    elif key=='06-11':
        mcb_months = [6,7,8,9,10,11]
    elif key=='06-08':
        mcb_months = [6,7,8]
    elif key=='09-02':
        mcb_months = [1,2,9,10,11,12]
    elif key=='09-11':
        mcb_months = [9,10,11]
    elif key=='12-02':
        mcb_months = [1,2,12]       
    mcb_days_list = []
    for month in mcb_months:
        mcb_days_list.append(month_day_dict[month])
    total_ship_op_cost[key] = (sum(mcb_days_list)*n_ships*daily_ship_op_cost)/1e9 #billions of USD
    print(key,': ', round(total_ship_op_cost[key],3), 'billions of USD')

