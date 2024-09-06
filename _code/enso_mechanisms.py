### PURPOSE: Script to calculate ENSO metrics (SOI, Walker cell, thermocline, wind stress)
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/28/2024
### LAST MODIFIED: 09/06/2024

### NOTES: adapted from enso_metrics_v1.py

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
import matplotlib.dates as mdates
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import ticker
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import geo
import cartopy
from cartopy.mpl.patch import geos_to_path
import itertools
import mpl_toolkits.mplot3d
from matplotlib.collections import PolyCollection, LineCollection

plt.ion(); #uncomment for interactive plotting

dask.config.set({"array.slicing.split_large_chunks": False})


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


# Get list of control climatology ensemble members
clim_files =  glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/TS/b.e21.BSMYLE.f09_g17.1970-'+month_init+'*.nc')
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
atm_varnames_monthly_subset = ['LANDFRAC','TS','PRECT','PS','U','V']
ocn_varnames_monthly_subset = ['TEMP','TAUX']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day


## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS
atm_monthly_ctrl_clim_xr = fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*atm_clim_concat.nc')[0]))

# Compute ensemble climatogical mean from 1970-2014
ts_clim_ensemble_mean = atm_monthly_ctrl_clim_xr.TS.mean(dim=('member'))
ps_clim_ensemble_mean = atm_monthly_ctrl_clim_xr.PS.mean(dim=('member'))


# OCN
data_dir = '/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/ocn_tseries/TEMP/regrid/'
target_dir = os.path.join(data_dir,'processed')

# Read in ensemble and time averaged data
ocn_monthly_ctrl_clim_xr = fun.reorient_netCDF(xr.open_dataset(glob.glob(target_dir+'/r288x192*_v2.nc')[0]))['TEMP']
temp_clim_ensemble_mean = ocn_monthly_ctrl_clim_xr.mean(dim='member')


## READ IN CONTROL SIMULATION & PRE-PROCESS
# ATM
atm_monthly_ctrl={}
ts_ctrl_anom={}
ps_ctrl_anom={}
ts_ctrl_anom_std={}
ts_ctrl_anom_sem={}
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
    ##DRIFT CORRECTION
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_ctrl_anom[key] = atm_monthly_ctrl[key]['TS']*1
    ps_ctrl_anom[key] = atm_monthly_ctrl[key]['PS']*1
    for month in i_month:
        ts_ctrl_anom[key].loc[{'time':[t for t in pd.to_datetime(ts_ctrl_anom[key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean.sel(month=month)
        ps_ctrl_anom[key].loc[{'time':[t for t in pd.to_datetime(ps_ctrl_anom[key].time.values) if t.month==month]}]-=ps_clim_ensemble_mean.sel(month=month)
    ts_ctrl_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    ps_ctrl_anom[key].attrs['units']='Pa'
    # Compute standard deviation
    ts_ctrl_anom_std[key]=ts_ctrl_anom[key].std(dim='member')
    # Compute twice standard error
    ts_ctrl_anom_sem[key]=2 * ts_ctrl_anom[key].std(dim='member')/np.sqrt(len(ts_ctrl_anom[key].member))


# OCN
ocn_monthly_ctrl={}
temp_ctrl_anom={}
temp_ctrl_anom_std={}
temp_ctrl_anom_sem={}
ctrl_keys=['']

for key in ctrl_keys:
    ocn_monthly_ctrl_single_mem = {}
    # Experiment number
    exp_num = intersect_members[0][:-4]
    path = os.path.join('/_data/realtime/ocn_processed', exp_num)
    # Read in ensemble and time averaged data
    ocn_monthly_ctrl[key] = fun.reorient_netCDF(xr.open_dataset(path+'/b.e21.BSMYLE.f09_g17.2015-05.TEMP.'+intersect_members[0][-3:]+'-'+intersect_members[-1][-3:]+'.nc'))
    # Unit correction
    # Convert depth from cm to m
    ocn_monthly_ctrl[key]['z_t'] = ocn_monthly_ctrl[key]['z_t']/100
    ocn_monthly_ctrl[key]['z_t'].attrs['units'] = 'm'
    # Convert TAUX dyne/cm2 to Pa
    ocn_monthly_ctrl[key]['TAUX'] = ocn_monthly_ctrl[key]['TAUX']*(100**2)
    ocn_monthly_ctrl[key]['TAUX'].attrs['units'] = 'Pa'
    ##DRIFT CORRECTION
    # By month climatology
    i_month=np.arange(1,13,1)
    temp_ctrl_anom[key] = ocn_monthly_ctrl[key]['TEMP']*1
    for month in i_month:
        temp_ctrl_anom[key].loc[{'time':[t for t in pd.to_datetime(temp_ctrl_anom[key].time.values) if t.month==month]}]-=temp_clim_ensemble_mean.sel(month=month)
    temp_ctrl_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    # Compute standard deviation
    temp_ctrl_anom_std[key]=temp_ctrl_anom[key].std(dim='member')
    # Compute twice standard error
    temp_ctrl_anom_sem[key]=2 * temp_ctrl_anom[key].std(dim='member')/np.sqrt(len(temp_ctrl_anom[key].member))



## READ IN MCB SIMULATIONS & PRE-PROCESS
# ATM
atm_monthly_mcb={}
ts_mcb_anom={}
ps_mcb_anom={}
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
    ##DRIFT CORRECTION
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_mcb_anom[key] = atm_monthly_mcb[key]['TS']*1
    ps_mcb_anom[key] = atm_monthly_mcb[key]['PS']*1
    for month in i_month:
        ts_mcb_anom[key].loc[{'time':[t for t in pd.to_datetime(ts_mcb_anom[key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean.sel(month=month)
        ps_mcb_anom[key].loc[{'time':[t for t in pd.to_datetime(ps_mcb_anom[key].time.values) if t.month==month]}]-=ps_clim_ensemble_mean.sel(month=month)
    ts_mcb_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    ps_mcb_anom[key].attrs['units']='Pa'
    # Compute standard deviation
    ts_mcb_anom_std[key]=ts_mcb_anom[key].std(dim='member')
    # Compute twice standard error
    ts_mcb_anom_sem[key]=2 * ts_mcb_anom[key].std(dim='member')/np.sqrt(len(ts_mcb_anom[key].member))


# OCN
ocn_monthly_mcb={}
temp_mcb_anom={}
temp_mcb_anom_std={}
temp_mcb_anom_sem={}
for key in mcb_keys:
    ocn_monthly_mcb_single_mem = {}
    # Experiment number
    exp_num = mcb_sims[key][0][:-4]
    path = os.path.join('/_data/MCB/ocn_processed', exp_num)
    # Read in ensemble and time averaged data
    ocn_monthly_mcb[key] = fun.reorient_netCDF(xr.open_dataset(path+'/b.e21.BSMYLE.f09_g17.2015-05.TEMP.'+mcb_sims[key][0][-3:]+'-'+mcb_sims[key][-1][-3:]+'.nc'))
    # Unit correction
    # Convert depth from cm to m
    ocn_monthly_mcb[key]['z_t'] = ocn_monthly_mcb[key]['z_t']/100
    ocn_monthly_mcb[key]['z_t'].attrs['units'] = 'm'
    # Convert TAUX dyne/cm2 to Pa
    ocn_monthly_mcb[key]['TAUX'] = ocn_monthly_mcb[key]['TAUX']*(100**2)
    ocn_monthly_mcb[key]['TAUX'].attrs['units'] = 'Pa'
    ##DRIFT CORRECTION
    # By month climatology
    i_month=np.arange(1,13,1)
    temp_mcb_anom[key] = ocn_monthly_mcb[key]['TEMP']*1
    for month in i_month:
        temp_mcb_anom[key].loc[{'time':[t for t in pd.to_datetime(temp_mcb_anom[key].time.values) if t.month==month]}]-=temp_clim_ensemble_mean.sel(month=month)
    temp_mcb_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    # Compute standard deviation
    temp_mcb_anom_std[key]=temp_mcb_anom[key].std(dim='member')
    # Compute twice standard error
    temp_mcb_anom_sem[key]=2 * temp_mcb_anom[key].std(dim='member')/np.sqrt(len(temp_mcb_anom[key].member))


#%% COMPUTE ANOMALIES FOR SELECT VARIABLES
## MONTHLY ATMOSPHERE
# Create empty dictionaries for anomalies
atm_monthly_anom = {}
atm_monthly_ensemble_anom = {}

## Loop through subsetted varnames list. 
print('##ATM MONTHLY##')
for key in mcb_keys:
    print(key)
    atm_monthly_anom[key] = {}
    atm_monthly_ensemble_anom[key] = {}
    for varname in atm_varnames_monthly_subset:
        print(varname)
        atm_monthly_anom[key][varname] = atm_monthly_mcb[key][varname] - atm_monthly_ctrl[ctrl_keys[0]][varname]
        atm_monthly_anom[key][varname].attrs['units'] = atm_monthly_ctrl[ctrl_keys[0]][varname].units
        atm_monthly_ensemble_anom[key][varname] = atm_monthly_anom[key][varname].mean(dim='member')
        atm_monthly_ensemble_anom[key][varname].attrs['units'] = atm_monthly_ctrl[ctrl_keys[0]][varname].units

## MONTHLY OCEAN
# Create empty dictionaries for anomalies
ocn_monthly_anom = {}
ocn_monthly_ensemble_anom = {}

## Loop through subsetted varnames list. 
print('##OCN MONTHLY##')
for key in mcb_keys:
    print(key)
    ocn_monthly_anom[key] = {}
    ocn_monthly_ensemble_anom[key] = {}
    for varname in ocn_varnames_monthly_subset:
        print(varname)
        ocn_monthly_anom[key][varname] = ocn_monthly_mcb[key][varname] - ocn_monthly_ctrl[ctrl_keys[0]][varname]
        ocn_monthly_anom[key][varname].attrs['units'] = ocn_monthly_ctrl[ctrl_keys[0]][varname].units
        ocn_monthly_ensemble_anom[key][varname] = ocn_monthly_anom[key][varname].mean(dim='member')
        ocn_monthly_ensemble_anom[key][varname].attrs['units'] = ocn_monthly_ctrl[ctrl_keys[0]][varname].units


## RETRIEVE AND GENERATE ANALYSIS AREA MASKS
# Get overlay mask files (area is the same for all of them so can just pick one)
seeding_mask = fun.reorient_netCDF(xr.open_dataset('/_data/sesp_mask_CESM2_0.9x1.25_v3.nc'))

# Force seeding mask lat, lon to equal the output CESM2 data (rounding errors)
seeding_mask = seeding_mask.assign_coords({'lat':atm_monthly_ctrl[ctrl_keys[0]]['lat'], 'lon':atm_monthly_ctrl[ctrl_keys[0]]['lon']})
# Subset 1 month of seeded grid cells 
seeding_mask_seed = seeding_mask.mask.isel(time=9)
# Add cyclical point for ML 
seeding_mask_seed_wrap, lon_wrap = add_cyclic_point(seeding_mask_seed,coord=seeding_mask_seed.lon)


## Define Niño 3.4 region
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
        

## Define Walker Circulation Strength Index regions
# Central/Eastern Pacific
lat_max = 5
lat_min = -5
lon_max = -80
lon_min = -160
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
cepac_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
cepac_mask_wrap, lon_wrap = add_cyclic_point(cepac_mask,coord=cepac_mask.lon)

# Indian Ocean/Western Pacific
lat_max = 5
lat_min = -5
lon_max = 160
lon_min = 80
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
iowpac_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
iowpac_mask_wrap, lon_wrap = add_cyclic_point(iowpac_mask,coord=iowpac_mask.lon)


## Define thermocline depth region
# Central Tropical Pacific
lat_max = 10
lat_min = -10
lon_max = -110
lon_min = -160
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
ctp_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)

# Eastern equatorial Pacific SOI
lat_max = 5
lat_min = -5
lon_max = -80
lon_min = -130
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
eep_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)

# Western equatorial Pacific SOI
lat_max = 5
lat_min = -5
lon_max = 140
lon_min = 90
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
wep_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)

## Define Timmerman region
lat_max = 2
lat_min = -2
lon_WP_max = -80
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 120
# Generate Niño 4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
tim_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
tim_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

tim_mask = tim_WP_mask + tim_EP_mask

# Eastern equatorial Pacific Timmerman
lat_max = 2
lat_min = -2
lon_max = -80
lon_min = -130
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
etim_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)


# Thermocline slope index
# W. Pacific (-2 to 2; 160 to -150)
lat_max = 2
lat_min = -2
lon_WP_max = -150
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 160
# Generate Niño 4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
wtherm_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
wtherm_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

wtherm_mask = wtherm_WP_mask + wtherm_EP_mask
# E. Pacific (-2 to 2; -140 to -90)
lat_max = 2
lat_min = -2
lon_max = -90
lon_min = -140
# Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].TEMP.isel(member=0, time=0)*0
etherm_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)


#%% CALCULATE SOI
# Define function
def calc_soi(in_xr):
    """
    Function to calculate ensemble mean Southern Oscillation Index.
    Returns time series of SOI values.
    :param in_xr: datarray with dimensions [member, time, lat, lon] to calculate index
    """
    # Define coordinates for Darwin and Tahiti
    lat_dar = -12.46
    lon_dar = 130.84
    lat_tah = -17.65
    lon_tah = -149.5

    ps_dar =in_xr.sel(lat=lat_dar, lon=lon_dar, method='nearest')
    ps_tah = in_xr.sel(lat=lat_tah, lon=lon_tah, method='nearest')
    ps_diff_std = (ps_tah-ps_dar).std(dim='time')
    soi = ((ps_tah-ps_dar)/ps_diff_std).mean(dim='member')
    return soi

# Calculate SOI
soi_ctrl = calc_soi(ps_ctrl_anom[''])
soi_mcb={}
soi_djf_mcb={}
for key in mcb_keys:
    soi_mcb[key]= calc_soi(ps_mcb_anom[key])
    if month_init=='05':
        soi_djf_mcb[key] = round(float((soi_mcb[key].isel(time=slice(7,10))).mean(dim='time').values),3)

#%% CALCULATE WALKER CIRCULATION STRENGTH INDEX
# Control - Climatology
cepac_ctrl = fun.calc_weighted_mean_tseries(ps_ctrl_anom[''].where(cepac_mask>0,drop=True))
iowpac_ctrl = fun.calc_weighted_mean_tseries(ps_ctrl_anom[''].where(iowpac_mask>0,drop=True))
walker_index_ctrl = (cepac_ctrl-iowpac_ctrl).mean(dim='member')
walker_index_ctrl_sem = 2*(cepac_ctrl-iowpac_ctrl).std(dim='member')/np.sqrt(len(ps_ctrl_anom[''].member))
walker_index_ctrl_lower_plot = walker_index_ctrl-walker_index_ctrl_sem
walker_index_ctrl_upper_plot = walker_index_ctrl+walker_index_ctrl_sem
# MCB - Climatology
walker_index_anom = {}
walker_index_djf_anom={}
for key in mcb_keys:
    cepac_mcb = fun.calc_weighted_mean_tseries(ps_mcb_anom[key].where(cepac_mask>0,drop=True))
    iowpac_mcb = fun.calc_weighted_mean_tseries(ps_mcb_anom[key].where(iowpac_mask>0,drop=True))
    walker_index_anom[key] = (cepac_mcb-iowpac_mcb).mean(dim='member') 
    if month_init=='05':
        walker_index_djf_anom[key] = round(float((walker_index_anom[key].isel(time=slice(7,10))).mean(dim='time').values),2)

#%% CALCULATE THERMOCLINE SLOPE INDEX
# Set depth interpolation level to 500 m
z_t_interp = np.arange(0,500)
## Compute HISTORICAL Z20
# W. Pacific
thermocline_mask=wtherm_mask
temp_historical_subset = temp_clim_ensemble_mean.where(thermocline_mask>0,drop=True).interpolate_na(dim='lon',fill_value='extrapolate')
wpac_z20_historical = np.abs(temp_historical_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
# E. Pacific
thermocline_mask=etherm_mask
temp_historical_subset = temp_clim_ensemble_mean.where(thermocline_mask>0,drop=True).interpolate_na(dim='lon',fill_value='extrapolate')
epac_z20_historical = np.abs(temp_historical_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
z20_historical_tseries = fun.calc_weighted_mean_tseries(wpac_z20_historical)-fun.calc_weighted_mean_tseries(epac_z20_historical)
## Compute CONTROL Z20
# W. Pacific
thermocline_mask=wtherm_mask
temp_ctrl_subset = ((ocn_monthly_ctrl[''].TEMP).where(thermocline_mask>0,drop=True)).interpolate_na(dim='lon',fill_value='extrapolate')
wpac_z20_ctrl = np.abs(temp_ctrl_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
# E. Pacific
thermocline_mask=etherm_mask
temp_ctrl_subset = ((ocn_monthly_ctrl[''].TEMP).where(thermocline_mask>0,drop=True)).interpolate_na(dim='lon',fill_value='extrapolate')
epac_z20_ctrl = np.abs(temp_ctrl_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
z20_ctrl_tseries = fun.calc_weighted_mean_tseries(wpac_z20_ctrl)-fun.calc_weighted_mean_tseries(epac_z20_ctrl)
z20_ctrl_clim_anom = z20_ctrl_tseries.astype(float)*1
for month in i_month:
    z20_ctrl_clim_anom.loc[{'time':[t for t in pd.to_datetime(z20_ctrl_clim_anom.time.values) if t.month==month]}]-=z20_historical_tseries.sel(month=month)
z20_ctrl_anom = z20_ctrl_clim_anom
z20_ctrl_sem = 2*z20_ctrl_anom.std(dim='member')/np.sqrt(len(z20_ctrl_anom.member))
z20_ctrl_anom_lower_plot=z20_ctrl_anom.mean(dim='member')-z20_ctrl_sem
z20_ctrl_anom_upper_plot=z20_ctrl_anom.mean(dim='member')+z20_ctrl_sem
z20_anom_df = z20_ctrl_anom.mean(dim='member').to_dataframe().reset_index()
z20_anom_df['experiment']='Control'
## Compute MCB CTP Z20 ANOMALY
for key in mcb_keys:
    print(key)
    # Mask out Z20 lat, lon with mask
    # W. Pacific
    thermocline_mask=wtherm_mask
    temp_mcb_subset = ((ocn_monthly_mcb[key].TEMP.mean(dim=('member'))).where(thermocline_mask>0,drop=True)).interpolate_na(dim='lon',fill_value='extrapolate')
    wpac_z20_mcb = np.abs(temp_mcb_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
    # E. Pacific
    thermocline_mask=etherm_mask
    temp_mcb_subset = ((ocn_monthly_mcb[key].TEMP.mean(dim=('member'))).where(thermocline_mask>0,drop=True)).interpolate_na(dim='lon',fill_value='extrapolate')
    epac_z20_mcb = np.abs(temp_mcb_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')    
    z20_mcb_tseries = fun.calc_weighted_mean_tseries(wpac_z20_mcb)-fun.calc_weighted_mean_tseries(epac_z20_mcb)
    # z20 = z_t_interp[z20]
    # Calculate Z20 anomaly
    z20_mcb_clim_anom = z20_mcb_tseries.astype(float)*1
    for month in i_month:
        z20_mcb_clim_anom.loc[{'time':[t for t in pd.to_datetime(z20_mcb_clim_anom.time.values) if t.month==month]}]-=z20_historical_tseries.sel(month=month)
    z20_anom_tseries = z20_mcb_clim_anom
    # z20_anom_tseries = fun.calc_weighted_mean_tseries(z20_mcb_tseries - z20_ctrl_tseries)
    z20_anom_tseries_single_df = z20_anom_tseries.to_dataframe().drop(columns=['member']).reset_index()
    z20_anom_tseries_single_df['experiment'] = key
    z20_anom_df = pd.concat([z20_anom_df, z20_anom_tseries_single_df]).reset_index(drop=True)

# Get DJF mean Z20
z20_djf_anom_df = (z20_anom_df[(z20_anom_df['time']=='2015-12-16') |(z20_anom_df['time']=='2016-01-16') | (z20_anom_df['time']=='2016-02-13')].groupby(['experiment']).mean()).reset_index()


#%% CALCULATE EQUATORIAL WIND STRESS ANOMALIES
taux_eq_anom = {}
for key in mcb_keys:
    taux_eq_anom[key] = fun.calc_weighted_mean_tseries(ocn_monthly_anom[key]['TAUX'].where(tim_mask.isel(z_t=0)>0,drop=True)).mean(dim='member')


#%% CALCULATE MEAN SURFACE TEMPERATURE IN SESP AND NIÑO3.4 BOX
sesp_ts_mcb = {}
nino34_ts_mcb = {}
nino34_djf_anom_ts_mcb = {}
for key in mcb_keys:
    sesp_ts_mcb[key] = fun.calc_weighted_mean_tseries(atm_monthly_mcb[key].TS.mean(dim='member').where(seeding_mask_seed>0,drop=True))
    nino34_ts_mcb[key] = fun.calc_weighted_mean_tseries(atm_monthly_mcb[key].TS.mean(dim='member').where(nino34_mask>0,drop=True))
    if month_init=='05':
        nino34_djf_anom_ts_mcb[key] = fun.calc_weighted_mean_tseries(ts_mcb_anom[key].isel(time=slice(7,10)).mean(dim=('time','member')).where(nino34_mask>0,drop=True))
sesp_ts_ctrl = fun.calc_weighted_mean_tseries(atm_monthly_ctrl[''].TS.mean(dim='member').where(seeding_mask_seed>0,drop=True))
nino34_ts_ctrl = fun.calc_weighted_mean_tseries(atm_monthly_ctrl[''].TS.mean(dim='member').where(nino34_mask>0,drop=True))
if month_init=='05':
    nino34_djf_anom_ts_ctrl = fun.calc_weighted_mean_tseries(ts_ctrl_anom[''].isel(time=slice(7,10)).mean(dim=('time','member')).where(nino34_mask>0,drop=True))
# Calc SEM
sesp_ts_ctrl_sem = fun.calc_weighted_mean_tseries(2*(atm_monthly_ctrl[''].TS).std(dim='member').where(seeding_mask_seed>0,drop=True)/np.sqrt(len(ts_ctrl_anom[''].member)))
sesp_ts_ctrl_lower_plot = sesp_ts_ctrl-sesp_ts_ctrl_sem
sesp_ts_ctrl_upper_plot = sesp_ts_ctrl+sesp_ts_ctrl_sem
nino34_ts_ctrl_sem = fun.calc_weighted_mean_tseries(2*(ts_ctrl_anom['']).std(dim='member').where(nino34_mask>0,drop=True)/np.sqrt(len(ts_ctrl_anom[''].member)))
nino34_ts_ctrl_lower_plot = nino34_ts_ctrl-nino34_ts_ctrl_sem
nino34_ts_ctrl_upper_plot = nino34_ts_ctrl+nino34_ts_ctrl_sem


# Plot time series
mcb_colors = {'':'#a50f15','06-02':'#a50f15','06-08':'#a50f15','06-11':'#a50f15','09-02':'#ef3b2c','09-11':'#ef3b2c','12-02':'#fc9272'} # reds=start month
mcb_linestyle = {'':'solid','06-02':'solid','06-08':(0, (1, 1)),'06-11':'dashed','09-02':'dashed','09-11':(0, (1, 1)),'12-02':(0, (1, 1))} # linestyle=duration
# Subset MCB window
mcb_on_start_dict = {'06-02':1,'06-08':1,'06-11':1,'09-02':4,'09-11':4,'12-02':7}
mcb_on_end_dict = {'06-02':10,'06-08':4,'06-11':7,'09-02':10,'09-11':7,'12-02':10}

fig = plt.figure(figsize=(6,8),layout='constrained')
spec = fig.add_gridspec(5,1)
## WALKER CELL STRENGTH
ax0 = fig.add_subplot(spec[0:2, 0])
# Control
# PLOT 2 STANDARD ERRORS
plt.fill_between(walker_index_ctrl.time, walker_index_ctrl_lower_plot, walker_index_ctrl_upper_plot,color='k', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(walker_index_ctrl.time,walker_index_ctrl,color='k',linewidth=3,label='Control');
for key in mcb_keys:
    # PLOT ENSEMBLE MEAN
    plt.plot(walker_index_anom[key].time,walker_index_anom[key],color=mcb_colors[key],linestyle=mcb_linestyle[key],linewidth=3,label='MCB '+key);
# PLOT AESTHETICS
plt.ylim(-400,200); #2015
# plt.ylim(-350,250); #1997
plt.axhline(0,linestyle='--',color='k');
# PLOT PEAK ENSO DJF
# Subset first year of simulation
t1=ts_ctrl_anom[''].isel(time=slice(4,16))
# Subset DJF and rename by month
tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
plt.fill_between(tslice.time, plt.gca().get_ylim()[0]-10, plt.gca().get_ylim()[1]+10,color='steelblue', alpha=0.2);
# Format dates
ax=plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_xticklabels([])
plt.ylabel(r'$\Delta$Walker cell strength (Pa)', fontsize=12);
plt.title('a', fontsize=14, fontweight='bold',loc='left');

## THERMOCLINE SLOPE
ax1 = fig.add_subplot(spec[2:4, 0])
# Control
# PLOT 2 STANDARD ERRORS
plt.fill_between(z20_ctrl_anom_lower_plot.time, z20_ctrl_anom_lower_plot, z20_ctrl_anom_upper_plot,color='k', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(z20_anom_df.loc[z20_anom_df['experiment']=='Control']['time'], z20_anom_df.loc[z20_anom_df['experiment']=='Control']['TEMP'],linestyle='solid', c='k',linewidth=3,label='Control');
# MCB
# mcb_keys = ['06-02','12-02'] #UNCOMMENT FOR 06-02 and 12-02 experiments only
for key in mcb_keys:
    # PLOT ENSEMBLE MEAN
   plt.plot(z20_anom_df.loc[z20_anom_df['experiment']==key]['time'], z20_anom_df.loc[z20_anom_df['experiment']==key]['TEMP'],linestyle=mcb_linestyle[key], c=mcb_colors[key],linewidth=3,label='MCB '+key);
# PLOT AESTHETICS
plt.ylim(-45,35); #2015
# plt.ylim(-50,55); #1997
plt.axhline(0,linestyle='--',color='k');
# PLOT PEAK ENSO DJF
# Subset first year of simulation
t1=ts_ctrl_anom[''].isel(time=slice(4,16))
# Subset DJF and rename by month
tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
plt.fill_between(tslice.time, plt.gca().get_ylim()[0]-10, plt.gca().get_ylim()[1]+10,color='steelblue', alpha=0.2);
# Format dates
ax=plt.gca()
xbounds=ax.get_xlim();
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
plt.xlabel('Time',fontsize=12); 
plt.ylabel('Thermocline slope index (m)', fontsize=12);
plt.title('b', fontsize=14, fontweight='bold',loc='left');

### Add legend 
ax4 = fig.add_subplot(spec[4, 0])
mcb_legend_y = {'06-02':6,'06-08':4,'06-11':5,'09-02':3,'09-11':2,'12-02':1}
mcb_legend_label = {'06-02':'Jun-Feb','06-08':'Jun-Aug','06-11':'Jun-Nov','09-02':'Sep-Feb','09-11':'Sep-Nov','12-02':'Dec-Feb'}
mcb_legend_longname = {'06-02':'Full effort','06-08':'Early action','06-11':'','09-02':'','09-11':'','12-02':'11th hour'}
ax4.set_xlim(xbounds);
ax4.set_ylim(0.5,6.5);
for key in mcb_keys:
    xmin = ts_mcb_anom[key].time.isel(time=mcb_on_start_dict[key]).values
    xmax = ts_mcb_anom[key].time.isel(time=mcb_on_end_dict[key]-1).values
    plt.hlines(mcb_legend_y[key], xmin=xmin,xmax=xmax,
            color=mcb_colors[key],linestyle=mcb_linestyle[key],linewidth=3)
    plt.scatter([xmin,xmax],[mcb_legend_y[key],mcb_legend_y[key]],
                color=mcb_colors[key],s=100)
    plt.annotate(mcb_legend_label[key],xy=(pd.to_datetime(xmax)+datetime.timedelta(days=40) ,mcb_legend_y[key]-.2),color=mcb_colors[key],fontsize=12)
    if year_init=='2015':
        plt.annotate(mcb_legend_longname[key],xy=(datetime.datetime(2016, 12, 15) ,mcb_legend_y[key]-.2),color=mcb_colors[key],fontsize=12)
    elif year_init=='1997':
        plt.annotate(mcb_legend_longname[key],xy=(datetime.datetime(1998, 12, 15) ,mcb_legend_y[key]-.2),color=mcb_colors[key],fontsize=12)
plt.ylabel('MCB strategy', fontsize=12);
ax4.set_xticks([]);ax4.set_yticks([]);
plt.setp(ax4.spines.values(), color=None);



#%%  3 PANEL FIG 3: SLP, SST, Z20 PLOTS
## DJF SLP bar graph
# Subset raw SLP for each region for Walker index
# CLIMATOLOGY
t1 = ps_clim_ensemble_mean
slp_cepac_clim= fun.calc_weighted_mean_tseries(t1.where(cepac_mask>0,drop=True)).loc[{'month':[t for t in t1.month.values if (t==12)|(t==1)|(t==2)]}].mean().values/100
slp_iowpac_clim = fun.calc_weighted_mean_tseries(t1.where(iowpac_mask>0,drop=True)).loc[{'month':[t for t in t1.month.values if (t==12)|(t==1)|(t==2)]}].mean().values/100
# CONTROL
t1 = atm_monthly_ctrl[''].PS.mean(dim='member').isel(time=slice(4,16))
slp_cepac_control= fun.calc_weighted_mean_tseries(t1.where(cepac_mask>0,drop=True)).loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].mean().values/100
slp_iowpac_control= fun.calc_weighted_mean_tseries(t1.where(iowpac_mask>0,drop=True)).loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].mean().values/100
# FULL EFFORT MCB
t1 = atm_monthly_mcb['06-02'].PS.mean(dim='member').isel(time=slice(4,16))
slp_cepac_fullmcb= fun.calc_weighted_mean_tseries(t1.where(cepac_mask>0,drop=True)).loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].mean().values/100
slp_iowpac_fullmcb= fun.calc_weighted_mean_tseries(t1.where(iowpac_mask>0,drop=True)).loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].mean().values/100
# Combine SLPs into dataframe
regions = ('Indo/Western Pacific','Central/Eastern Pacific')
slp_values = {'Reference': (float(slp_iowpac_clim),float(slp_cepac_clim)),
            'El Niño':(float(slp_iowpac_control),float(slp_cepac_control)),
            'Full effort MCB':(float(slp_iowpac_fullmcb),float(slp_cepac_fullmcb)),}

# Create barplot
x = np.arange(len(regions))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0
cmap = {'Reference':'lightgray','El Niño':'#f4a582','Full effort MCB':'#92c5de'}
# cmap = {'Reference':'lightgray','El Niño':'#f4a582','Full effort MCB':'white'}

fig, ax = plt.subplots(figsize=(10,3),layout='constrained')
for attribute, measurement in slp_values.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=cmap[attribute])
    #ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Sea-level pressure (hPa)',fontsize=14);
# ax.set_title('a', fontsize=14, fontweight='bold',loc='left');
ax.tick_params(labelsize=12);
ax.legend(loc='upper left',fontsize=14);
ax.set_ylim(100000/100, 101200/100);
ax.annotate('Indo/Western Pacific', xy = [-0.08,100250/100],color='k',fontweight='bold',size=14);
ax.annotate('Central/Eastern Pacific', xy = [.9,101050/100],color='k',fontweight='bold',size=14);
#ax.axis('off');
ax.set(xticklabels=[]);ax.tick_params(bottom=False);
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)


## DJF TS contour map
# CLIMATOLOGY
t1 = ts_clim_ensemble_mean
ts_clim_plot = fun.weighted_temporal_mean_clim(t1.loc[{'month':[t for t in t1.month.values if (t==12)|(t==1)|(t==2)]}])
# CONTROL
t1 = ts_ctrl_anom[''].mean(dim='member').isel(time=slice(4,16))
ts_control_plot = fun.weighted_temporal_mean_clim(t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].groupby('time.month').mean())
landmask = atm_monthly_ctrl[''].LANDFRAC.isel(member=0,time=0)
ts_control_plot = xr.where(landmask<0.1,ts_control_plot,np.nan )
# FULL EFFORT MCB
t1 = ts_mcb_anom['06-02'].mean(dim='member').isel(time=slice(4,16))
ts_fullmcb_plot = fun.weighted_temporal_mean_clim(t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].groupby('time.month').mean())
ts_fullmcb_plot = xr.where(landmask<0.1,ts_fullmcb_plot,np.nan )

## CREATE 3D map
# Remove white line for plotting over Pacific Ocean.
plot_proj = ccrs.PlateCarree(central_longitude=180)
# Control contourf
ts_control_plot_reorient = fun.reorient_netCDF(ts_control_plot,target=360)
ts_control_plot_reorient=ts_control_plot_reorient.where((ts_control_plot_reorient.lat<=25)&(ts_control_plot_reorient.lat>=-25)&(ts_control_plot_reorient.lon<=285)&(ts_control_plot_reorient.lon>=75),drop=True)
lat = ts_control_plot_reorient.lat
lon = ts_control_plot_reorient.lon
lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
# MCB contour
ts_fullmcb_plot_reorient = fun.reorient_netCDF(ts_fullmcb_plot,target=360)
ts_fullmcb_plot_reorient=ts_fullmcb_plot_reorient.where((ts_fullmcb_plot_reorient.lat<=25)&(ts_fullmcb_plot_reorient.lat>=-25)&(ts_fullmcb_plot_reorient.lon<=285)&(ts_fullmcb_plot_reorient.lon>=75),drop=True)
# Crop extent for thermocline region
tim_plot_reorient = fun.reorient_netCDF(tim_mask,target=360)
tim_plot_reorient=tim_plot_reorient.where((tim_plot_reorient.lat<=25)&(tim_plot_reorient.lat>=-25)&(tim_plot_reorient.lon<=285)&(tim_plot_reorient.lon>=75),drop=True)
# Crop extent for Walker regions
iowpac_plot_reorient = fun.reorient_netCDF(iowpac_mask,target=360)
iowpac_plot_reorient=iowpac_plot_reorient.where((iowpac_plot_reorient.lat<=25)&(iowpac_plot_reorient.lat>=-25)&(iowpac_plot_reorient.lon<=285)&(iowpac_plot_reorient.lon>=75),drop=True)
cepac_plot_reorient = fun.reorient_netCDF(cepac_mask,target=360)
cepac_plot_reorient=cepac_plot_reorient.where((cepac_plot_reorient.lat<=25)&(cepac_plot_reorient.lat>=-25)&(cepac_plot_reorient.lon<=285)&(cepac_plot_reorient.lon>=75),drop=True)


# Create figure
fig = plt.figure(figsize=(10,10))
ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')
# Make an axes that we can use for mapping the data in 2d.
proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=plot_proj)

# Create control TS contourf 
levels = np.linspace(-4,4,21)
cs = proj_ax.contourf(lon, lat, ts_control_plot_reorient,cmap='RdBu_r',vmin=-3,vmax=3,levels=levels, extend='both', transform=ccrs.PlateCarree())
# Create MCB TS contours
cs2 = proj_ax.contour(lon, lat, ts_fullmcb_plot_reorient,levels=levels,cmap='RdBu_r',linewidths=2, transform=ccrs.PlateCarree())
# Create Thermocline region
cs3 = proj_ax.contour(tim_plot_reorient.lon,tim_plot_reorient.lat,tim_plot_reorient.isel(z_t=0), transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k',linestyles='--',linewidths=1.5)
# Create Walker regions
cs4 = proj_ax.contour(iowpac_plot_reorient.lon,iowpac_plot_reorient.lat,iowpac_plot_reorient, transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k',linewidths=1.5)
cs5 = proj_ax.contour(cepac_plot_reorient.lon,cepac_plot_reorient.lat,cepac_plot_reorient, transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k',linewidths=1.5)

# Add 2D geocontours to 3D map
ax3d.projection = proj_ax.projection
fun.add_contourf3d(ax3d, cs)
fun.add_contour3d(ax3d, cs2)
fun.add_contour3d(ax3d, cs3)
fun.add_contour3d(ax3d, cs4)
fun.add_contour3d(ax3d, cs5)

# Use the convenience (private) method to get the extent as a shapely geometry.
clip_geom = proj_ax._get_extent_geom().buffer(0)
# Add land features to the bottom z level
zbase = 0
fun.add_feature3d(ax3d, cartopy.feature.LAND,clip_geom=clip_geom, zs=zbase)

# Change axis limits
ax3d.set_xlim(-90, 90)
ax3d.set_ylim(-13, 13)
ax3d.set_zlim(-1, 1)

# Close the intermediate (2d) figure
plt.close(proj_ax.figure);

# Set view
ax3d.view_init(elev=10., azim=-90);
# Turn off grid
plt.grid(False);
plt.axis('off');
# Add colorbar
plt.colorbar(cs,orientation='vertical',ax = ax3d, label='Temperature (\N{DEGREE SIGN}C)', extend='both',shrink=0.3,pad=0.001);


## DJF Z20 as a function of latitude and depth
#%% CALCULATE EQUATORIAL TROPICAL PACIFIC Z20 ANOMALY
# Pick which region you want to calculate thermocline depth over
thermocline_mask = tim_mask
# Set depth interpolation level to 500 m
z_t_interp = np.arange(0,500)
## Compute HISTORICAL Z20
temp_historical_subset = temp_clim_ensemble_mean.where(thermocline_mask>0,drop=True).interpolate_na(dim='lon',fill_value='extrapolate')
z20_historical = np.abs(temp_historical_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
# z20_historical_tseries = z20_historical.groupby('time.month').mean()
z20_historical_tseries = fun.weighted_temporal_mean_clim(z20_historical.loc[{'month':[t for t in z20_historical.month.values if (t==12)|(t==1)|(t==2)]}]).mean(dim=('lat'))
z20_historical_tseries = fun.reorient_netCDF(z20_historical_tseries,target=360)
## Compute CONTROL Z20
temp_ctrl_subset = ((ocn_monthly_ctrl[''].TEMP).where(thermocline_mask>0,drop=True)).interpolate_na(dim='lon',fill_value='extrapolate')
z20_ctrl = np.abs(temp_ctrl_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
t1 = z20_ctrl.isel(time=slice(4,16))
z20_ctrl_tseries = fun.weighted_temporal_mean_clim(t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].groupby('time.month').mean()).mean(dim=('member','lat'))
z20_ctrl_tseries = fun.reorient_netCDF(z20_ctrl_tseries,target=360)
## Compute MCB CTP Z20 
# Mask out Z20 lat, lon with mask
temp_mcb_subset = ((ocn_monthly_mcb['06-02'].TEMP.mean(dim=('member'))).where(thermocline_mask>0,drop=True)).interpolate_na(dim='lon',fill_value='extrapolate')
z20_mcb = np.abs(temp_mcb_subset.interp(z_t=z_t_interp)-20).argmin(dim='z_t')
t1 = z20_mcb.isel(time=slice(4,16))
z20_mcb_tseries = fun.weighted_temporal_mean_clim(t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}].groupby('time.month').mean()).mean(dim=('lat'))
z20_mcb_tseries = fun.reorient_netCDF(z20_mcb_tseries,target=360)

# Create z20 line plot of depth vs. longitude
fig, ax = plt.subplots(figsize=(10,3),layout='constrained')
ax.plot(z20_historical_tseries.lon,z20_historical_tseries.values,linewidth=4,color=cmap['Reference'],label='Reference')
ax.plot(z20_ctrl_tseries.lon,z20_ctrl_tseries.values,linewidth=4,color=cmap['El Niño'],label='El Niño')
ax.plot(z20_mcb_tseries.lon,z20_mcb_tseries.values,linewidth=4,color=cmap['Full effort MCB'],label='Full effort MCB')
ax.vlines(120,ymin=30,ymax=220,linewidth=3,linestyle='dashed',color='k')
ax.vlines(280,ymin=30,ymax=220,linewidth=3,linestyle='dashed',color='k')
ax.set_ylabel('Depth (m)',fontsize=14);
ax.invert_yaxis();
plt.ylim(220,30);plt.xlim(75,285);
ax.set_xticks([90,120,150,180,210,240,270]);
ax.set_xticklabels(['90°E','120°E','150°E','180°E', '150°W', '120°W','90°W']);
ax.tick_params(labelsize=12);
ax.set_xlabel('Longitude (degrees)',fontsize=14);


# Create ED table of ENSO indicators
columns=['MCB_strategy','Niño3.4_SST','SOI','Walker_strength_index','Thermocline_slope_index']
mcb_exp_reordered = ['06-02','06-11','06-08','09-02','09-11','12-02']
enso_indicator_df = pd.DataFrame(columns=columns)
for key in mcb_exp_reordered:
    nino34_djf_anom = float(nino34_djf_anom_ts_mcb[key]-nino34_djf_anom_ts_ctrl)
    df_append = pd.DataFrame([[key, round(nino34_djf_anom,3),round(soi_djf_mcb[key],3),round(walker_index_djf_anom[key],3), round(float(z20_djf_anom_df[z20_djf_anom_df['experiment']==key]['TEMP'].values),3)]],columns=columns)
    enso_indicator_df = pd.concat([enso_indicator_df,df_append])
enso_indicator_df = enso_indicator_df