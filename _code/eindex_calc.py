### PURPOSE: Script to calculate E/C indices from surface temperature EOFs
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 08/27/2024
### LAST MODIFIED: 09/06/2024

### Note: script adapted from smyle_fosi_eindex_v3.py
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

# Optional setup
plt.ion();
dask.config.set({"array.slicing.split_large_chunks": False})
#run this line in console before starting ipython if getting OMP_NUM_THREADS error
os.environ["OMP_NUM_THREADS"] = "1"

##################################################################################################################
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


# # Get interesction of control and MCB ensemble members so we only keep members that are in both
intersect_members = ctrl_members[0:len(mcb_members)]


# Create variable subset list
atm_varnames_monthly_subset = ['TS']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day


## READ IN CONTROL SMYLE-FOSI HISTORICAL SIMULATIONS
data_dir='/_data/SMYLE-FOSI/regrid/'
# Read in potential temperature and select surface layer only to reduce file sice
ocn_temp_hist_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(data_dir+'r288x192.g.e22.GOMIPECOIAF_JRA-1p4-2018.TL319_g17.SMYLE.005.pop.h.TEMP.030601-036812.nc'))).TEMP.isel(z_t=0)
# Reassign time values to be between 1958-2020 (Yeager et al., 2022)
start_yr=1958
delta_yr = start_yr - int(ocn_temp_hist_xr.time[0].dt.year.values)
ocn_temp_hist_xr = ocn_temp_hist_xr.assign_coords(time=ocn_temp_hist_xr['time']+datetime.timedelta(days=365*delta_yr))
ocn_temp_hist_xr = ocn_temp_hist_xr.assign_coords(time=ocn_temp_hist_xr.indexes['time'].to_datetimeindex())
# Fix any rounding errors for lat,lon grid so they match SMYLE exactly
sample_path = '/_data/SMYLE-MCB/realtime/b.e21.BSMYLE.f09_g17.2015-05.001/atm/proc/tseries/month_1/b.e21.BSMYLE.f09_g17.2015-05.001.cam.h0.TS.201505-201704.nc'
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
    ts_mcb_diff[key] = (ts_mcb_anom[key].mean(dim='member')-ts_ctrl_anom[''].mean(dim='member'))
    ts_mcb_anom_corrected[key] = ts_hist_anom+ts_mcb_diff[key]
    ts_mcb_anom_corrected[key].attrs['units'] = '\N{DEGREE SIGN}C'


#%% CREATE INDEX MASKS
# Get overlay mask files (area is the same for all of them so can just pick one)
seeding_mask = fun.reorient_netCDF(xr.open_dataset('/_data/mask_CESM/sesp_mask_CESM2_0.9x1.25_v3.nc'))

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


### CALCULATE E/C INDEX
# Define number of modes you want to calculate EOFs
nmodes=2

## Historical EOF analysis (DJF ONLY)
# Mask E/C index region
ts_hist_detrend_enso = ts_hist_detrend_djf.where(ecindex_mask>0,drop=True)
# Calculate Niño 3 index time series for historical
ts_hist_detrend_nino3 = fun.calc_weighted_mean_tseries(ts_hist_detrend_djf.where(nino3_mask>0,drop=True))
# Comput EOFs and PCS
eof_solver = Eof(ts_hist_detrend_enso)
ts_hist_eofs = eof_solver.eofs(neofs=nmodes)
ts_hist_vf = eof_solver.varianceFraction(neigs=nmodes)
ts_hist_pcs = eof_solver.pcs(pcscaling=1,npcs=nmodes)
# Scale PC1 sign to be positively correlated with Nino3
for i in np.arange(0,nmodes,1):
    corrcoef = np.corrcoef(ts_hist_pcs[:,i].values,ts_hist_detrend_nino3.values)
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
    ts_hist_eofs[i,:,:] = ts_hist_eofs[i,:,:].values*scaling
    ts_hist_pcs[:,i] = ts_hist_pcs[:,i].values*scaling
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
    # Mask E/C index region
    ts_mcb_detrend_enso = ts_mcb_detrend_djf[key].where(ecindex_mask>0,drop=True)
    # Calculate Niño 3 index time series for historical
    ts_mcb_detrend_nino3 = fun.calc_weighted_mean_tseries(ts_mcb_detrend_djf[key].where(nino3_mask>0,drop=True))
    # Comput EOFs and PCS
    eof_solver = Eof(ts_mcb_detrend_enso)
    ts_mcb_eofs[key] = eof_solver.eofs(neofs=nmodes)
    ts_mcb_vf[key] = eof_solver.varianceFraction(neigs=nmodes)
    ts_mcb_pcs[key] = eof_solver.pcs(pcscaling=1,npcs=nmodes)
    # Scale PC1 sign to be positively correlated with Nino3
    for i in np.arange(0,nmodes,1):
        # Want to compute correlation with Nino3 without MCB simulation
        corrcoef = np.corrcoef(ts_mcb_pcs[key].isel(time=slice(None,len(ts_hist_detrend_djf.time)))[:,i].values,ts_mcb_detrend_nino3.isel(time=slice(None,len(ts_hist_detrend_djf.time))).values)
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
        ts_mcb_eofs[key][i,:,:] = ts_mcb_eofs[key][i,:,:].values*scaling
        ts_mcb_pcs[key][:,i] = ts_mcb_pcs[key][:,i].values*scaling
    # Calculate E and C indices
    pc1 = ts_mcb_pcs[key].sel(mode=0)
    pc2 = ts_mcb_pcs[key].sel(mode=1)
    ts_mcb_eindex[key] = (pc1-pc2)/(np.sqrt(2))
    ts_mcb_cindex[key] = (pc1+pc2)/(np.sqrt(2))


## Plot EOF comparisons for historical and appended MCB timeseries
version_num = 1
plot_labels = ['a','b','c','d','e','f','g','h','j','k','l','m','n','o']
mcb_legend_label = {'06-02':'Jun-Feb','06-08':'Jun-Aug','06-11':'Jun-Nov','09-02':'Sep-Feb','09-11':'Sep-Nov','12-02':'Dec-Feb'}
fig=plt.figure(figsize=(6,14));
subplot_num=1
for m in ts_hist_eofs.mode.values:
    plt.subplot(7,2,subplot_num);
    p=plt.imshow(fun.reorient_netCDF(ts_hist_eofs.sel(mode=m),target=360),vmin=-0.035,vmax=0.035,cmap='RdBu_r');#plt.colorbar(orientation='vertical',label='SST (\N{DEGREE SIGN}C)');
    plt.annotate('Historical (Mode='+str(m)+')', xy=(.14,1.03),fontsize=12, xycoords='axes fraction',color='k');
    plt.title(plot_labels[subplot_num-1],fontweight='bold',fontsize=12,loc='left');
    plt.xticks([]);plt.yticks([]);
    subplot_num+=1
for key in mcb_keys:
    for m in ts_hist_eofs.mode.values:
        plt.subplot(7,2,subplot_num);
        plt.imshow(fun.reorient_netCDF(ts_mcb_eofs[key].sel(mode=m),target=360),vmin=-0.035,vmax=0.035,cmap='RdBu_r');
        plt.annotate(mcb_legend_label[key]+' MCB (Mode='+str(m)+')',fontsize=12, xy=(.14,1.03), xycoords='axes fraction',color='k');
        plt.title(plot_labels[subplot_num-1],fontweight='bold',fontsize=12,loc='left');
        plt.xticks([]);plt.yticks([]);
        subplot_num+=1
plt.tight_layout();
fig.subplots_adjust(bottom=0.15, top=0.97,hspace=.1);
cbar_ax = fig.add_axes([0.13, 0.12, 0.75, 0.02]) #rect kwargs [left, bottom, width, height];
fig.colorbar(p, cax=cbar_ax,orientation='horizontal', label='SST (\N{DEGREE SIGN}C)',pad=0.15);



## MCB
mcb_df = pd.DataFrame()
for key in mcb_keys:
    print(key)
    # Create dataframe from output
    # Convert all xarrays into dataframes
    pc1_df =  ts_mcb_pcs[key].isel(mode=0).to_dataframe().reset_index()
    pc1_df = pc1_df.rename(columns={'pcs':'PC1'}).drop(columns='mode')
    pc2_df =  ts_mcb_pcs[key].isel(mode=1).to_dataframe().reset_index()
    pc2_df = pc2_df.rename(columns={'pcs':'PC2'}).drop(columns='mode')
    eindex_df = ts_mcb_eindex[key].to_dataframe().reset_index().rename(columns={'pcs':'E-index'})
    cindex_df = ts_mcb_cindex[key].to_dataframe().reset_index().rename(columns={'pcs':'C-index'})
    # Merge on time
    tmp_df = pd.concat([pc1_df,pc2_df.drop(columns='time'),eindex_df.drop(columns='time'),cindex_df.drop(columns='time')],axis=1)
    tmp_df['exp_id'] = key
    # Add extra column indicating whether value is an MCB year or not (last 2 appended years are MCB)
    tmp_list = ['N']*len(ts_hist_pcs.time)
    tmp_list.extend(['Y','Y'])
    tmp_df['MCB_year'] = tmp_list
    mcb_df = pd.concat([mcb_df, tmp_df],ignore_index=True)


## HISTORICAL
# Create dataframe from output
# Convert all xarrays into dataframes
pc1_df =  ts_hist_pcs.isel(mode=0).to_dataframe().reset_index()
pc1_df = pc1_df.rename(columns={'pcs':'PC1'}).drop(columns='mode')
pc2_df =  ts_hist_pcs.isel(mode=1).to_dataframe().reset_index()
pc2_df = pc2_df.rename(columns={'pcs':'PC2'}).drop(columns='mode')
eindex_df = ts_hist_eindex.to_dataframe().reset_index().rename(columns={'pcs':'E-index'})
cindex_df = ts_hist_cindex.to_dataframe().reset_index().rename(columns={'pcs':'C-index'})
# Merge on time
hist_df = pd.concat([pc1_df,pc2_df.drop(columns='time'),eindex_df.drop(columns='time'),cindex_df.drop(columns='time')],axis=1)
# Historical is being treated as Control for this analysis
hist_df['exp_id'] = 'Control'
hist_df['MCB_year'] = 'N'

# Expore PCS and E and C index as csv
ecindex_df = pd.concat([mcb_df, hist_df],ignore_index=True)

# Plot scatter of E/C indices
# Subset DJF add ENSO peak
if year_init=='1997':
    peak_yrs = [1997,1998]
elif year_init=='2015':
    peak_yrs = [2015,2016]

version_num = 1
mcb_colors = {'Control':'k','06-02':'#a50f15','06-08':'#a50f15','06-11':'#a50f15','09-02':'#ef3b2c','09-11':'#ef3b2c','12-02':'#fc9272'} # reds=start month
mcb_marker =  {'Control':'o','06-02':'*','06-08':'^','06-11':'P','09-02':'P','09-11':'^','12-02':'^'} # duration
mcb_legend_label = {'06-02':'Jun-Feb','06-08':'Jun-Aug','06-11':'Jun-Nov','09-02':'Sep-Feb','09-11':'Sep-Nov','12-02':'Dec-Feb'}
mcb_linestyle = {'Control':'solid','06-02':'solid','06-08':(0, (1, 1)),'06-11':'dashed','09-02':'dashed','09-11':(0, (1, 1)),'12-02':(0, (1, 1))} # linestyle=duration
plt.figure(figsize=(8,6));
# # CONTROL
key='Control'
plt.scatter(ts_hist_eindex, ts_hist_cindex,c='grey',marker='o',alpha=0.25,s=75,label='Historical');
plt.scatter(ts_hist_eindex.sel(time=peak_yrs[1]), ts_hist_cindex.sel(time=peak_yrs[1]),c='k',marker='o',s=150,label=key);
for key in mcb_keys:
    # hard coded to take second to last value which is the MCB El Niño peak
    plt.scatter(ts_mcb_eindex[key].isel(time=-2), ts_mcb_cindex[key].isel(time=-2),c=mcb_colors[key],marker=mcb_marker[key],label='MCB '+mcb_legend_label[key],s=150);
plt.xlabel('E-index (s.d.)',fontsize=12);
plt.ylabel('C-index (s.d.)',fontsize=12);
plt.grid(linestyle='--');
plt.axvline(0,linewidth=2,c='k');plt.axhline(0,linewidth=2,c='k');
plt.annotate(year_init+' El Niño',fontsize=12, xy=(.4,1.03), xycoords='axes fraction',color='k');
if year_init == '2015':
    plt.title('a',fontweight='bold',fontsize=14,loc='left');
    plt.legend(loc='lower right');
elif year_init == '1997':
    plt.title('b',fontweight='bold',fontsize=14,loc='left');
