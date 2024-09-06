### PURPOSE: Script to define ENSO temperature and precipitation regional impacts
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/28/2024
### LAST MODIFIED: 09/06/2024

### NOTES: adapted from enso_regional_impact_mask.py

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
import cartopy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.ion(); #uncomment for interactive plotting

##################################################################################################################
## WHICH EXPERIMENT ARE YOU READING IN? ##
month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
## UNCOMMENT THESE OPTIONS FOR DEMO ##
month_init = '05'
##################################################################################################################

## READ IN DATA
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


# Create variable subset list
atm_varnames_monthly_subset = ['TS','PRECT']


## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS
atm_monthly_ctrl_clim_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*TS_PRECT_concat.nc')[0])))


## COMPUTE LONG TERM STANDARD DEVIATION AND MONTHLY CLIMATOLOGY MEAN FROM 1970-2014
# Subset time from 1970-2014
hist_ext = atm_monthly_ctrl_clim_xr.isel(time=atm_monthly_ctrl_clim_xr['time.year']<2015)
# Compute climatogical mean from 1970-2014
ts_clim_ensemble_mean = hist_ext.TS.mean(dim=('member')).groupby('time.month').mean() # By monthly climatology
prect_clim_ensemble_mean = hist_ext.PRECT.mean(dim=('member')).groupby('time.month').mean() # By monthly climatology


# Subset data before AND including 2015 El Niño event
atm_monthly_hist_subset = atm_monthly_ctrl_clim_xr.isel(time=atm_monthly_ctrl_clim_xr['time.year']<=2016)
# Subset DJF (during peak ENSO) and JJA (post-peak ENSO)
atm_monthly_hist_subset_djf = atm_monthly_hist_subset.where((atm_monthly_hist_subset.time.dt.month==1)|(atm_monthly_hist_subset.time.dt.month==2)|(atm_monthly_hist_subset.time.dt.month==12),drop=True)
atm_monthly_hist_subset_jja = atm_monthly_hist_subset.where((atm_monthly_hist_subset.time.dt.month==6)|(atm_monthly_hist_subset.time.dt.month==7)|(atm_monthly_hist_subset.time.dt.month==8),drop=True)


#%% How well does the historical SMYLE reproduce EL Niños?
# Get overlay mask files (area is the same for all of them so can just pick one)
seeding_mask = fun.reorient_netCDF(xr.open_dataset('/_data/sesp_mask_CESM2_0.9x1.25_v3.nc'))

# Force seeding mask lat, lon to equal the output CESM2 data (rounding errors)
seeding_mask = seeding_mask.assign_coords({'lat':atm_monthly_hist_subset['lat'], 'lon':atm_monthly_hist_subset['lon']})
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
zeros_mask = atm_monthly_hist_subset.TS.isel(member=0, time=0)*0
nino34_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
nino34_mask_wrap, lon_wrap = add_cyclic_point(nino34_mask,coord=nino34_mask.lon)

## CALCULATE ENSEMBLE MEAN STANDARD DEVIATION FOR ALL GRID CELLS
# DJF 
hist_ensemble_djf_std = atm_monthly_hist_subset_djf[['TS','PRECT']].std(dim='time').mean(dim='member')
# JJA
hist_ensemble_jja_std = atm_monthly_hist_subset_jja[['TS','PRECT']].std(dim='time').mean(dim='member')

## CALCULATE ANOMALY FOR THE 4 STRONGEST EL NINOS WITHIN 45 YEAR SPAN (1972, 1982, 1997, 2015)
# Subset major historical El Nino events
nino_events = [1972, 1982, 1991, 1997, 2015] # major

for n in range(len(nino_events)):
    print(nino_events[n])
    nino_single_djf = atm_monthly_hist_subset_djf.where(((atm_monthly_hist_subset_djf.time.dt.year==nino_events[n])&(atm_monthly_hist_subset_djf.time.dt.month==12))|\
                                                        ((atm_monthly_hist_subset_djf.time.dt.year==nino_events[n]+1)&(atm_monthly_hist_subset_djf.time.dt.month==1))|\
                                                        ((atm_monthly_hist_subset_djf.time.dt.year==nino_events[n]+1)&(atm_monthly_hist_subset_djf.time.dt.month==2)),drop=True)
    nino_single_jja = atm_monthly_hist_subset_jja.where(((atm_monthly_hist_subset_jja.time.dt.year==nino_events[n]+1)&(atm_monthly_hist_subset_jja.time.dt.month==6))|\
                                                        ((atm_monthly_hist_subset_jja.time.dt.year==nino_events[n]+1)&(atm_monthly_hist_subset_jja.time.dt.month==7))|\
                                                        ((atm_monthly_hist_subset_jja.time.dt.year==nino_events[n]+1)&(atm_monthly_hist_subset_jja.time.dt.month==8)),drop=True)
    if n==0:
        nino_djf_composite = nino_single_djf
        nino_jja_composite = nino_single_jja
    elif n>0:
        nino_djf_composite=xr.concat([nino_djf_composite,nino_single_djf],dim='time')
        nino_jja_composite=xr.concat([nino_jja_composite,nino_single_jja],dim='time')


# Calculate average anomaly
# DJF
nino_djf_composite_ts_anom = nino_djf_composite.TS.mean(dim=('member','time')) - atm_monthly_hist_subset_djf.TS.mean(dim=('member','time'))
nino_djf_composite_prect_anom = nino_djf_composite.PRECT.mean(dim=('member','time')) - atm_monthly_hist_subset_djf.PRECT.mean(dim=('member','time'))
# JJA
nino_jja_composite_ts_anom = nino_jja_composite.TS.mean(dim=('member','time')) - atm_monthly_hist_subset_jja.TS.mean(dim=('member','time'))
nino_jja_composite_prect_anom = nino_jja_composite.PRECT.mean(dim=('member','time')) - atm_monthly_hist_subset_jja.PRECT.mean(dim=('member','time'))


# Set significance threshold and plot
# EDIT THIS LINE
sig_threshold = 0.1
##################################################################################################################################################################
# Mask anomalies by significance threshold
# DJF
sig_djf_ts = xr.where(np.abs(nino_djf_composite_ts_anom)>sig_threshold*hist_ensemble_djf_std.TS, nino_djf_composite_ts_anom, np.nan)
sig_djf_prect = xr.where(np.abs(nino_djf_composite_prect_anom)>sig_threshold*hist_ensemble_djf_std.PRECT, nino_djf_composite_prect_anom, np.nan)
# JJA
sig_jja_ts = xr.where(np.abs(nino_jja_composite_ts_anom)>sig_threshold*hist_ensemble_jja_std.TS, nino_jja_composite_ts_anom, np.nan)
sig_jja_prect = xr.where(np.abs(nino_jja_composite_prect_anom)>sig_threshold*hist_ensemble_jja_std.PRECT, nino_jja_composite_prect_anom, np.nan)
# Remove white line if plotting over Pacific Ocean.
# Create a reference grid (1x1)
lat_new = np.arange(-90, 91, 1)
lon_new = np.arange(-180., 181., 1)


# DJF
# TS
x=sig_djf_ts.interp(lat=lat_new, lon=lon_new, method='linear', kwargs={'fill_value': 'extrapolate'})
lat = x.lat
lon = x.lon
data = x
data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
plot_proj = ccrs.Robinson(central_longitude=180)
plt.figure(figsize=(6,9));
ax = plt.subplot(2,1,1, projection=plot_proj,transform=plot_proj)
p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal','label':'Temperature ('+nino_djf_composite.TS.units+')','fraction':0.075});
ax.coastlines(); p.axes.set_global();
plt.title('a',fontweight='bold',fontsize=14,loc='left');
# PRECT
x=sig_djf_prect.interp(lat=lat_new, lon=lon_new, method='linear', kwargs={'fill_value': 'extrapolate'})
lat = x.lat
lon = x.lon
data = x
data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
plot_proj = ccrs.Robinson(central_longitude=180)
ax = plt.subplot(2,1,2, projection=plot_proj,transform=plot_proj)
p = x.plot.contourf(ax=ax,vmin=-8,vmax=8,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal','label':'Precipitation ('+nino_djf_composite.PRECT.units+')','fraction':0.075});
ax.coastlines(); p.axes.set_global();
plt.title('b',fontweight='bold',fontsize=14,loc='left');
plt.tight_layout();


#%% MAKE MASK OF 1's and 0's for warm/cold and wet/dry
djf_ts_warm = xr.where(sig_djf_ts>0, 1, 0)
djf_ts_warm = djf_ts_warm.rename('Warm')
djf_ts_cold = xr.where(sig_djf_ts<0, 1, 0)
djf_ts_cold = djf_ts_cold.rename('Cold')
djf_prect_wet = xr.where(sig_djf_prect>0, 1, 0)
djf_prect_wet = djf_prect_wet.rename('Wet')
djf_prect_dry = xr.where(sig_djf_prect<0, 1, 0)
djf_prect_dry = djf_prect_dry.rename('Dry')

# PLOT MASKS
# DJF warm
x = djf_ts_warm
plt.figure(figsize=(8,6));
ax = plt.subplot(2,2,1, projection=plot_proj,transform=plot_proj)
p = x.plot.contourf(ax=ax,vmin=0,vmax=1,cmap='viridis',transform= ccrs.PlateCarree(),levels=3,cbar_kwargs={'orientation':'horizontal'});
ax.coastlines(color='grey'); p.axes.set_global();
plt.title('DJF warm');
# DJF cold
x = djf_ts_cold
ax = plt.subplot(2,2,2, projection=plot_proj,transform=plot_proj)
p = x.plot.contourf(ax=ax,vmin=0,vmax=1,cmap='viridis',transform= ccrs.PlateCarree(),levels=3,cbar_kwargs={'orientation':'horizontal'});
ax.coastlines(color='grey'); p.axes.set_global();
plt.title('DJF cold');
# DJF wet
x = djf_prect_wet
ax = plt.subplot(2,2,3, projection=plot_proj,transform=plot_proj)
p = x.plot.contourf(ax=ax,vmin=0,vmax=1,cmap='viridis',transform= ccrs.PlateCarree(),levels=3,cbar_kwargs={'orientation':'horizontal'});
ax.coastlines(color='grey'); p.axes.set_global();
plt.title('DJF wet');
# DJF dry
x = djf_prect_dry
ax = plt.subplot(2,2,4, projection=plot_proj,transform=plot_proj)
p = x.plot.contourf(ax=ax,vmin=0,vmax=1,cmap='viridis',transform= ccrs.PlateCarree(),levels=3,cbar_kwargs={'orientation':'horizontal'});
ax.coastlines(color='grey'); p.axes.set_global();
plt.title('DJF dry');
# Plotting aesthetics
plt.suptitle(str(sig_threshold)+'\u03C3');
plt.tight_layout();

# Save masks
djf_mask_combined = xr.merge([djf_ts_warm, djf_ts_cold, djf_prect_wet, djf_prect_dry])
# Uncomment to save intermediate output
djf_mask_combined.to_netcdf('/_data/enso_regions/djf_major_nino_regions_'+str(month_init)+'_'+str(sig_threshold)+'_sigma_v2.nc',mode='w',format='NETCDF4')



##################################################################################################################
#%% CHECKPOINT: REGIONAL IMPACTS MAP + BAR PLOT
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
import cartopy
from matplotlib.lines import Line2D
plt.ion();

dask.config.set({"array.slicing.split_large_chunks": False})

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
# EDIT THIS LINE
sig_threshold = 0.1
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

atm_varnames_monthly_subset = ['TS','PRECT']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day

## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS
atm_monthly_ctrl_clim_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*TS_PRECT_concat.nc')[0])))


## COMPUTE LONG TERM STANDARD DEVIATION AND MONTHLY CLIMATOLOGY MEAN FROM 1970-2014
# Subset time from 1970-2014
hist_ext = atm_monthly_ctrl_clim_xr.isel(time=atm_monthly_ctrl_clim_xr['time.year']<2015)
# Compute climatogical mean from 1970-2014
ts_clim_ensemble_mean = hist_ext.TS.mean(dim=('member')).groupby('time.month').mean() # By monthly climatology
prect_clim_ensemble_mean = hist_ext.PRECT.mean(dim=('member')).groupby('time.month').mean() # By monthly climatology


## READ IN CONTROL SIMULATION & PRE-PROCESS
# ATM
atm_monthly_ctrl={}
ts_ctrl_anom={}
prect_ctrl_anom={}
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
    prect_ctrl_anom[key] = atm_monthly_ctrl[key]['PRECT']*1
    for month in i_month:
        ts_ctrl_anom[key].loc[{'time':[t for t in pd.to_datetime(ts_ctrl_anom[key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean.sel(month=month)
        prect_ctrl_anom[key].loc[{'time':[t for t in pd.to_datetime(prect_ctrl_anom[key].time.values) if t.month==month]}]-=prect_clim_ensemble_mean.sel(month=month)
    ts_ctrl_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    prect_ctrl_anom[key].attrs['units']='mm/day'
    # Compute standard deviation
    ts_ctrl_anom_std[key]=ts_ctrl_anom[key].std(dim='member')
    # Compute twice standard error
    ts_ctrl_anom_sem[key]=2 * ts_ctrl_anom[key].std(dim='member')/np.sqrt(len(ts_ctrl_anom[key].member))



## READ IN MCB SIMULATIONS & PRE-PROCESS
# ATM
atm_monthly_mcb={}
ts_mcb_anom={}
prect_mcb_anom={}
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
    prect_mcb_anom[key] = atm_monthly_mcb[key]['PRECT']*1
    for month in i_month:
        ts_mcb_anom[key].loc[{'time':[t for t in pd.to_datetime(ts_mcb_anom[key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean.sel(month=month)
        prect_mcb_anom[key].loc[{'time':[t for t in pd.to_datetime(prect_mcb_anom[key].time.values) if t.month==month]}]-=prect_clim_ensemble_mean.sel(month=month)
    ts_mcb_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    prect_mcb_anom[key].attrs['units']='mm/day'
    # Compute standard deviation
    ts_mcb_anom_std[key]=ts_mcb_anom[key].std(dim='member')
    # Compute twice standard error
    ts_mcb_anom_sem[key]=2 * ts_mcb_anom[key].std(dim='member')/np.sqrt(len(ts_mcb_anom[key].member))


## COMPUTE ANOMALIES FOR SELECT VARIABLES
## 1a) MONTHLY ATMOSPHERE
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


## RETRIEVE AND GENERATE ANALYSIS AREA MASKS
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
                      
# Read in ENSO impact region mask
djf_mask_combined=xr.open_dataset('/_data/enso_regions/djf_major_nino_regions_'+str(month_init)+'_'+str(sig_threshold)+'_sigma_v2.nc')
 
# Identify signficant cells (ensemble mean differences > 2*SE)
# Calculate standard error of control ensemble
atm_monthly_sig = {}
for key in mcb_keys:
    atm_monthly_sig[key] = {}
    for varname in atm_varnames_monthly_subset:
        print(varname)
        sem = stats.sem(atm_monthly_ctrl[ctrl_keys[0]][varname].values,axis=0)
        atm_monthly_sig[key][varname] = xr.where(np.abs(atm_monthly_ensemble_anom[key][varname])>2*np.abs(sem), 0,1)


# Calculate standard error of control ensemble for DJF of ENSO event
atm_djf_sig = {}
for key in mcb_keys:
    atm_djf_sig[key] = {}
    for varname in atm_varnames_monthly_subset:
        print(varname)
        # Subset first year of simulation
        t1=atm_monthly_ctrl[ctrl_keys[0]][varname].isel(time=slice(4,16))
        # Subset DJF and rename by month
        tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
        tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
        tslice = tslice.rename({'time':'month'})
        tslice = fun.weighted_temporal_mean_clim(tslice)
        sem = stats.sem(tslice.values,axis=0)
        # Subset MCB anomaly dataarray for DJF of first year
        t2=atm_monthly_ensemble_anom[key][varname].isel(time=slice(4,16))
        tslice2 =t2.loc[{'time':[t for t in pd.to_datetime(t2.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
        tslice2 =tslice2.assign_coords(time=pd.to_datetime(tslice2.time.values).month)
        tslice2 = tslice2.rename({'time':'month'})
        tslice2 = fun.weighted_temporal_mean_clim(tslice2)
        atm_djf_sig[key][varname] = xr.where(np.abs(tslice2)>2*np.abs(sem), 0,1)



# Define ENSO regions over which to calculate responses
# test = fun.weighted_temporal_mean(atm_monthly_ensemble_anom['06-02']['TS'].isel(time=slice(4,16)).loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]).mean(dim='time')
lat = djf_mask_combined.lat
lon = djf_mask_combined.lon

# a) S Asia warming
lat_min = -11
lat_max = 36
lon_min = 66
lon_max = 126
s_asia_warm = xr.where((djf_mask_combined.Warm>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
s_asia_warm = s_asia_warm.rename('a')
# # PLOT MASKS
# x = s_asia_warm
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# b) Japan warming
lat_min = 30
lat_max = 45
lon_min = 120
lon_max = 150
japan_warm = xr.where((djf_mask_combined.Warm>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
japan_warm = japan_warm.rename('b')
# # PLOT MASKS
# x = japan_warm
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# c) Alaska warming
lat_min = 50
lat_max = 72
lon_min = -170
lon_max = -90
alaska_warm = xr.where((djf_mask_combined.Warm>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
alaska_warm = alaska_warm.rename('c')
# # PLOT MASKS
# x = alaska_warm
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# d) SE U.S. cooling
lat_min = 20
lat_max = 50
lon_min = -121
lon_max = -75
se_us_cold = xr.where((djf_mask_combined.Cold>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
se_us_cold = se_us_cold.rename('d')
# # PLOT MASKS
# x = se_us_cold
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# e) S U.S. wettening
lat_min = 20
lat_max = 50
lon_min = -125
lon_max = -75
se_us_wet = xr.where((djf_mask_combined.Wet>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
se_us_wet = se_us_wet.rename('e')
# # PLOT MASKS
# x = se_us_wet
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# f) N Brazil drying
lat_min = -25
lat_max = 15
lon_min = -67
lon_max = -30
brazil_dry = xr.where((djf_mask_combined.Dry>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
brazil_dry = brazil_dry.rename('f')
# # PLOT MASKS
# x = brazil_dry
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# g) E Brazil warming
lat_min = -25
lat_max = 0
lon_min = -50
lon_max = -30
brazil_warm = xr.where((djf_mask_combined.Warm>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
brazil_warm = brazil_warm.rename('g')
# # PLOT MASKS
# x = brazil_warm
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# h) E Brazil wettening
lat_min = -30
lat_max = 0
lon_min = -50
lon_max = -34
brazil_wet = xr.where((djf_mask_combined.Wet>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
brazil_wet = brazil_wet.rename('h')
# # PLOT MASKS
# x = brazil_wet
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# i) C/E equatorial Pacific wettening
lat_min = -15
lat_max = 10
ep_lon_min = -180
ep_lon_max = -80
wp_lon_min = 125
wp_lon_max = 181
equator_wet = xr.where((djf_mask_combined.Wet>0)&(lat>lat_min)&(lat<lat_max)&(((lon>ep_lon_min)&(lon<ep_lon_max))|((lon>wp_lon_min)&(lon<wp_lon_max))), 1, np.nan)
equator_wet = equator_wet.rename('i')
#equator_wet = equator_wet.interp(lat=lat_new, lon=lon_new, method='linear', kwargs={'fill_value': 'extrapolate'})
# # PLOT MASKS
# x = equator_wet
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# j) S equatorial Pacific drying
lat_min = -50
lat_max = 0
ep_lon_min = -180
ep_lon_max = -162
wp_lon_min = 80
wp_lon_max = 181
s_pacific_dry = xr.where((djf_mask_combined.Dry>0)&(lat>lat_min)&(lat<lat_max)&(((lon>ep_lon_min)&(lon<ep_lon_max))|((lon>wp_lon_min)&(lon<wp_lon_max))), 1, np.nan)
#s_pacific_dry = s_pacific_dry.interp(lat=lat_new, lon=lon_new, method='linear', kwargs={'fill_value': 'extrapolate'})
s_pacific_dry = s_pacific_dry.rename('j')
# # PLOT MASKS
# x = s_pacific_dry
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# k) Australia warming
lat_min = -40
lat_max = -11
lon_min = 112
lon_max = 154
aus_warm = xr.where((djf_mask_combined.Warm>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
aus_warm = aus_warm.rename('k')
# # PLOT MASKS
# x = aus_warm
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# l) South Africa drying
lat_min = -36
lat_max = -15
lon_min = 10
lon_max = 36
s_africa_dry = xr.where((djf_mask_combined.Dry>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
s_africa_dry = s_africa_dry.rename('l')
# # PLOT MASKS
# x = s_africa_dry
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# m) South Africa warming
lat_min = -36
lat_max = -15
lon_min = 10
lon_max = 36
s_africa_warm = xr.where((djf_mask_combined.Warm>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
s_africa_warm = s_africa_warm.rename('m')
# # PLOT MASKS
# x = s_africa_warm
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='RdBu_r',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# n) East Africa wettening
lat_min = -11
lat_max = 10
lon_min = 23
lon_max = 41
e_africa_wet = xr.where((djf_mask_combined.Wet>0)&(lat>lat_min)&(lat<lat_max)&(lon>lon_min)&(lon<lon_max), 1, np.nan)
e_africa_wet = e_africa_wet.rename('n')
# # PLOT MASKS
# x = e_africa_wet
# plt.figure(figsize=(8,6));
# ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)
# p = x.plot.contourf(ax=ax,vmin=-3,vmax=3,cmap='BrBG',transform= ccrs.PlateCarree(),levels=9,cbar_kwargs={'orientation':'horizontal'});
# ax.coastlines(color='grey'); p.axes.set_global();

# Combine all regions into one master array
djf_regions_combined = xr.merge([s_asia_warm, japan_warm, alaska_warm, se_us_cold, se_us_wet, brazil_dry, brazil_warm, brazil_wet, equator_wet, s_pacific_dry, aus_warm, s_africa_dry, s_africa_warm, e_africa_wet])
# Create list of temperature impact regions
t_regions = ['a', 'b', 'c', 'd', 'g', 'k', 'm']
t_warm_regions = ['a', 'b', 'c', 'g', 'k','m']
t_cool_regions = ['d']
p_wet_regions = ['e', 'h', 'i', 'n' ]
p_dry_regions = ['f','j','l']


## Take area weighted mean of MCB response and historical climatology value in each target region
# Subset first year (hard coded for month_init==05)
# Only display bars for 06-02 and 12-02 for illustration
mcb_keys_sub = ['06-02','06-08','12-02']
percent_opt = input('percent or absolute?: ')
djf_region_df = pd.DataFrame()
# for key in mcb_keys:  #comment for plotting
for key in mcb_keys_sub: #uncomment for plotting  
    print(key)  
    for region in list(djf_regions_combined.keys()):
        if region in t_regions:
            xr_in = atm_monthly_ensemble_anom[key]['TS'].isel(time=slice(4,16))
            clim_xr_in = ts_ctrl_anom[''].mean(dim='member').isel(time=slice(4,16))
            clim_xr_in_sem = ts_ctrl_anom[''].std(dim='member').isel(time=slice(4,16))/np.sqrt(len(ts_ctrl_anom[''].member))
        elif region not in t_regions:
            xr_in = atm_monthly_ensemble_anom[key]['PRECT'].isel(time=slice(4,16))
            clim_xr_in = prect_ctrl_anom[''].mean(dim='member').isel(time=slice(4,16))
            clim_xr_in_sem = prect_ctrl_anom[''].std(dim='member').isel(time=slice(4,16))/np.sqrt(len(prect_ctrl_anom[''].member))
        # Get peak DJF and average over the three months
        xr_in_djf = fun.weighted_temporal_mean(xr_in.loc[{'time':[t for t in pd.to_datetime(xr_in.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]).mean(dim='time')
        clim_xr_in_djf = fun.weighted_temporal_mean(clim_xr_in.loc[{'time':[t for t in pd.to_datetime(clim_xr_in.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]).mean(dim='time')
        clim_xr_in_sem_djf = fun.weighted_temporal_mean(clim_xr_in_sem.loc[{'time':[t for t in pd.to_datetime(clim_xr_in_sem.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]).mean(dim='time')
        # Mask out by each region, calculate spatial mean, and save as pandas datafame
        xr_in_djf_region_mean = float(fun.calc_weighted_mean_tseries(xr.where(djf_regions_combined[region]==1, xr_in_djf, np.nan)))
        clim_xr_in_djf_region_mean = float(fun.calc_weighted_mean_tseries(xr.where(djf_regions_combined[region]==1, clim_xr_in_djf, np.nan)))
        clim_xr_in_sem_djf_region_mean = float(fun.calc_weighted_mean_tseries(xr.where(djf_regions_combined[region]==1, clim_xr_in_sem_djf, np.nan)))
        if percent_opt =='percent':
            # Set ymin and ymax for exceedance column
            ymin=-100
            ymax=100
            xr_in_djf_region_mean = xr_in_djf_region_mean/abs(clim_xr_in_djf_region_mean)
        new_row = pd.Series({'experiment':key, 'region':region, 'anom':xr_in_djf_region_mean,'sem':clim_xr_in_sem_djf_region_mean})
        djf_region_df = pd.concat([djf_region_df,new_row.to_frame().T],ignore_index=True)
        djf_region_df['Anom>SEM'] = np.where(np.abs(djf_region_df['anom'])>np.abs(djf_region_df['sem']),'**','')
        if percent_opt=='percent':
            djf_region_df['exceed'] = np.where(np.abs(djf_region_df['anom'])>(ymax/100),'**','')




## FIG 4. REGIONAL EL NINO IMPACTS COMBINED SUBPLOTS ##
# Set color and hatching preferences for bar chart
mcb_col_rev = ['#4d9221']
mcb_col_amp = ['#c51b7d']
mcb_hatch = {'06-02':None,'06-08':None,'12-02':None} # no hatch for 2 cases

# Set figure dimensions and grid
fig = plt.figure(figsize=(13, 8),layout='constrained')
spec = fig.add_gridspec(4, 5)

# Hard code the row and column index for each subplot in grid
row_vec = [0,0,0,0,0,1,2,3,3,3,3,3,2,1]
col_vec = [0,1,2,3,4,4,4,4,3,2,1,0,0,0]

# Make base map with El Niño regions
# Create a reference grid (1x1)
lat_new = np.arange(-90, 91, 1)
lon_new = np.arange(-180., 181., 1)
# ac_example = ['a','c']
plot_proj = ccrs.PlateCarree(central_longitude=160)
ax0 = fig.add_subplot(spec[1:3, 1:4], projection=plot_proj,transform=plot_proj)
for region in list(djf_regions_combined.keys()):
# for region in ac_example:
    x = djf_regions_combined.interp(lat=lat_new, lon=lon_new, method='linear', kwargs={'fill_value': 'extrapolate'})[region]
    xnonan = x.fillna(0)
    if region in t_warm_regions:
        col = '#b2182b'
    elif region in t_cool_regions:
        col='#2166ac'
    elif region in p_wet_regions:
        col = '#018571'
    elif region in p_dry_regions:
        col = '#a6611a'
    p =x.plot.contourf(ax=ax0,levels=np.linspace(0,1.1,2), colors=col,alpha=0.5, transform= ccrs.PlateCarree(),add_colorbar=False);
    xnonan.plot.contour(ax=ax0,levels=np.linspace(0,1.1,2), colors=col, linewidth=0.5, transform= ccrs.PlateCarree(),add_colorbar=False);
ax0.coastlines(color='grey'); p.axes.set_global();
ax0.add_feature(cartopy.feature.LAND, color='gainsboro');
ax0.add_feature(cartopy.feature.OCEAN, color='white');
ax0.add_feature(cartopy.feature.BORDERS, color='grey');
ax0.spines['geo'].set_edgecolor('white');
ax0.legend([Line2D([0], [0],color='#b2182b',alpha=0.5,linestyle='None',marker='s',markersize=12),\
            Line2D([0], [0],color='#2166ac',alpha=0.5,linestyle='None',marker='s',markersize=12),\
            Line2D([0], [0],color='#018571',alpha=0.5,linestyle='None',marker='s',markersize=12),\
            Line2D([0], [0],color='#a6611a',alpha=0.5,linestyle='None',marker='s',markersize=12),\
            Line2D([0], [0],color='#4d9221',alpha=1,linestyle='None',marker='s',markersize=12),\
            Line2D([0], [0],color='#c51b7d',alpha=1,linestyle='None',marker='s',markersize=12)],\
            ['Warm','Cool', 'Wet','Dry','MCB ameliorates El Niño effect','MCB exacerbates El Niño effect'],bbox_to_anchor =(0.5,0), loc='lower center',\
        ncol=3, fancybox=True, shadow=False,frameon=True,fontsize=12),;
# Annotate letters for each region
lab = 'a'
lab_lon = 85
lab_lat= 30
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon, lab_lon], [lab_lat - 2, lab_lat - 15],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'b'
lab_lon = 135
lab_lat= 50
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon, lab_lon], [lab_lat - 2, lab_lat - 15],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'c'
lab_lon = 210
lab_lat= 40
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon, lab_lon], [lab_lat + 8, lab_lat + 21],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'd'
lab_lon = 227
lab_lat= 35
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon+6, lab_lon+19], [lab_lat+5, lab_lat+5],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'e'
lab_lon = 300
lab_lat= 22
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon-6, lab_lon-19], [lab_lat+5, lab_lat+5],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'f'
lab_lon = 320
lab_lat= 15
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon-6, lab_lon-18], [lab_lat+2, lab_lat-10],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'g'
lab_lon = 339
lab_lat= -10
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon-6, lab_lon-17], [lab_lat+5, lab_lat+5],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'h'
lab_lon = 339
lab_lat= -37
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon-6, lab_lon-18], [lab_lat+5, lab_lat+10],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'i'
lab_lon = 180
lab_lat= 18
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon, lab_lon], [lab_lat - 2, lab_lat - 15],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'j'
lab_lon = 210
lab_lat= -45
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon-6, lab_lon-19], [lab_lat+5, lab_lat+5],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'k'
lab_lon = 130
lab_lat= -55
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon, lab_lon], [lab_lat + 10, lab_lat + 35],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'l'
lab_lon = 28
lab_lat= -55
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon, lab_lon], [lab_lat + 10, lab_lat + 21],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'm'
lab_lon = 0
lab_lat= -25
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon+8, lab_lon+22], [lab_lat+5, lab_lat+5],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );
lab = 'n'
lab_lon = 52
lab_lat= -3
ax0.text(lab_lon, lab_lat, lab, size=14, fontweight='bold',
         horizontalalignment='center',
         transform=ccrs.PlateCarree());
plt.plot([lab_lon-6, lab_lon-17], [lab_lat+5, lab_lat+5],
         color='k', linestyle='-',
         transform=ccrs.PlateCarree(),
         );

pos1 = ax0.get_position() # get the original position 
pos2 = [pos1.x0 + -0.07, pos1.y0+ -0.05 ,  pos1.width+0.1, pos1.height+0.1] 
ax0.set_position(pos2) # set a new position


# Bar plots for % of ENSO anomaly for each region
for region in list(djf_regions_combined.keys()):
    #plt.figure(figsize=(4,3));
    plot_val = djf_region_df[djf_region_df['region']==region]['anom']
    if region in t_warm_regions:
        ylab = '\N{GREEK CAPITAL LETTER DELTA}T (°C)'
        ymin = -0.8
        ymax= 0.35
        colormat=np.where(plot_val>0, mcb_col_amp,mcb_col_rev)
        col = '#b2182b'
    elif region in t_cool_regions:
        ylab = '\N{GREEK CAPITAL LETTER DELTA}T (°C)'
        ymin = -0.8
        ymax= 0.35
        colormat=np.where(plot_val<0, mcb_col_amp,mcb_col_rev)
        col='#2166ac'
    elif region in p_wet_regions:
        ylab = '\N{GREEK CAPITAL LETTER DELTA}P (mm/day)'
        ymin = -1.7
        ymax=0.8
        colormat=np.where(plot_val>0, mcb_col_amp,mcb_col_rev)
        col = '#018571'
    elif region in p_dry_regions:
        ylab = '\N{GREEK CAPITAL LETTER DELTA}P (mm/day)'
        ymin = -1.7
        ymax=0.8
        colormat=np.where(plot_val<0, mcb_col_amp,mcb_col_rev)     
        col = '#a6611a'
    ax = fig.add_subplot(spec[row_vec[list(djf_regions_combined.keys()).index(region)],col_vec[list(djf_regions_combined.keys()).index(region)]])
    if percent_opt=='percent':
        ax.bar(djf_region_df[djf_region_df['region']==region]['experiment'], djf_region_df[djf_region_df['region']==region]['anom']*100,color=colormat,hatch=list(mcb_hatch.values()));
        ymin=-100
        ymax=100
        ax.set_ylim(ymin,ymax)
        ax.set_yticks(np.arange(ymin,ymax+ymax/2,ymax/2))
        plt.yticks(fontsize=12);
        ax.set_ylabel("% of ENSO anomaly",fontsize=12);
        # Add arrow over bars that exceed the axis limits
        for bar in ax.patches:
            if abs(bar.get_height())>ymax:
                ax.annotate('',
                xy=((bar.get_x() + bar.get_width() / 2),75*np.sign(bar.get_height())), xycoords='data',
                xytext=((bar.get_x() + bar.get_width() / 2),50*np.sign(bar.get_height())), textcoords='data', 
                arrowprops=dict(arrowstyle='simple', connectionstyle="arc3",color=bar.get_facecolor(),lw=2),annotation_clip=False)
        # ADD ANNOTATION FOR REGION A EXPERIMENTS
        if region=='a':
            ax.annotate('Full', (ax.patches[0].get_x()+ax.patches[0].get_width()/2,50), ha='center',va='center',size=10,fontweight='bold',color='#a50f15');
            ax.annotate('effort', (ax.patches[0].get_x()+ax.patches[0].get_width()/2,30), ha='center',va='center',size=10,fontweight='bold',color='#a50f15');
            ax.annotate('Early', (ax.patches[1].get_x()+ax.patches[1].get_width()/2,50), ha='center',va='center',size=10,fontweight='bold',color='#a50f15');
            ax.annotate('action', (ax.patches[1].get_x()+ax.patches[1].get_width()/2,30), ha='center',va='center',size=10,fontweight='bold',color='#a50f15');
            ax.annotate('11th', (ax.patches[2].get_x()+ax.patches[2].get_width()/2,50), ha='center',va='center',size=10,fontweight='bold',color='#fc9272');
            ax.annotate('hour', (ax.patches[2].get_x()+ax.patches[2].get_width()/2,30), ha='center',va='center',size=10,fontweight='bold',color='#fc9272');
    elif percent_opt=='absolute':
        ax.bar(djf_region_df[djf_region_df['region']==region]['experiment'], djf_region_df[djf_region_df['region']==region]['anom'],color=colormat,hatch=list(mcb_hatch.values()));
        ax.set_ylim(ymin,ymax)
        ax.set_ylabel(ylab)
    ax.set_title(region,color='k',loc='left',fontsize=14,fontweight='bold')
    ax.set(xticklabels=[]);ax.tick_params(bottom=False);
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
