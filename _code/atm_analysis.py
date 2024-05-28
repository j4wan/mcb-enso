### PURPOSE: Script to analyze atm output from SMYLE-MCB
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/28/2024

### NOTES: adapted from smyle_mcb_preanalysis_v2.py

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
plt.ion();

dask.config.set({"array.slicing.split_large_chunks": False})

##################################################################################################################
## THIS SCRIPT READS IN ONE ENSEMBLE OF EXPERIMENTS AT A TIME. ##
## WHICH EXPERIMENT ARE YOU READING IN? ##
month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
year_init = input('Which initialization year are you reading in (1997, 2015, 2019?): ')
sensitivity_opt = input('Sensitivity run (y or n)?: ') # y for 05-1997 and 05-2015 only, else n
##################################################################################################################

## READ IN DATA
# Get list of control ensemble members
if year_init=='1997':
    yr_init = ['1996','1997']
    enso_phase = ['nino']
elif year_init=='2015':
    yr_init = ['2014','2015']
    enso_phase = ['nino']
elif year_init=='2019':
    yr_init = ['2019','2020']
    enso_phase = ['nina']
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
# intersect_members = sorted(list(set(ctrl_members).intersection(mcb_members)))
intersect_members = ctrl_members[0:len(mcb_members)]


# Create variable subset list
atm_varnames_monthly_subset = ['CLDLOW','FSNS','FSNTOA','LANDFRAC','PRECT','PS','SWCF','TREFHT','TS','PSL','U10','U','V']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day


## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS AND COMPUTE CLIMATOLOGY
# Read in each ensemble member as a continuous time series by taking mean of overlapping periods
# ATM
target_dir='/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed'
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
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
                    overlap_time=xr.merge([da_merged,next_file],compat='override',join='inner').time
                    da_merge_intersect = da_merged.where(da_merged.time==overlap_time)
                    next_file_intersect = next_file.where(next_file.time==overlap_time)
                    da_merged = xr.merge([da_merged,next_file],compat='override',join='outer')
                    da_merged.loc[{'time':[t for t in overlap_time.values]}] = (da_merge_intersect+next_file_intersect)/2
                    da_merged.loc[{'time':[t for t in da_merged.time.values if t>overlap_time.values[-1]]}] = next_file.loc[{'time':[t for t in next_file.time.values if t>overlap_time.values[-1]]}]
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
    atm_monthly_ctrl_clim_xr.to_netcdf(target_dir+'/BSMYLE.'+str(pd.to_datetime(atm_monthly_ctrl_clim_xr.time.values[0]).year)+'-'+str(pd.to_datetime(atm_monthly_ctrl_clim_xr.time.values[-1]).year)+'-'+month_init+'.atm_tseries_combined.nc',mode='w',format='NETCDF4')
else:
    atm_monthly_ctrl_clim_xr = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*atm_tseries_combined.nc')[0])))

# Compute climatogical mean from 1970-2014
tslice = atm_monthly_ctrl_clim_xr.TS.loc[{'time':[t for t in pd.to_datetime(atm_monthly_ctrl_clim_xr.time.values) if (t.year<2015)]}]
# ts_clim_ensemble_mean = tslice.mean(dim=('member','time')) # By annual climatology
ts_clim_ensemble_mean = tslice.mean(dim=('member')).groupby('time.month').mean() # By monthly climatology


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
    # By annual climatology
    # ts_ctrl_anom[key]=atm_monthly_ctrl[key]['TS']-ts_clim_ensemble_mean
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_ctrl_anom[key] = atm_monthly_ctrl[key]['TS']*1
    for month in i_month:
        ts_ctrl_anom[key].loc[{'time':[t for t in pd.to_datetime(ts_ctrl_anom[key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean.sel(month=month)
    ts_ctrl_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    # Compute standard deviation
    ts_ctrl_anom_std[key]=ts_ctrl_anom[key].std(dim='member')
    # Compute twice standard error
    ts_ctrl_anom_sem[key]=2 * ts_ctrl_anom[key].std(dim='member')/np.sqrt(len(ts_ctrl_anom[key].member))


## READ IN MCB SIMULATIONS & PRE-PROCESS
# ATM
atm_monthly_mcb={}
ts_mcb_anom={}
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
    # By annual climatology
    # ts_mcb_anom[key]=atm_monthly_mcb[key]['TS']-ts_clim_ensemble_mean
    # By month climatology
    i_month=np.arange(1,13,1)
    ts_mcb_anom[key] = atm_monthly_mcb[key]['TS']*1
    for month in i_month:
        ts_mcb_anom[key].loc[{'time':[t for t in pd.to_datetime(ts_mcb_anom[key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean.sel(month=month)
    ts_mcb_anom[key].attrs['units']='\N{DEGREE SIGN}C'
    # Compute standard deviation
    ts_mcb_anom_std[key]=ts_mcb_anom[key].std(dim='member')
    # Compute twice standard error
    ts_mcb_anom_sem[key]=2 * ts_mcb_anom[key].std(dim='member')/np.sqrt(len(ts_mcb_anom[key].member))


#%% COMPUTE ANOMALIES FOR SELECT VARIABLES
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



#%% PLOT ANOMALY MAPS
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
                      
# Define Niño 4 region
lat_max = 5
lat_min = -5
lon_WP_max = -150
lon_WP_min = -180
lon_EP_max = 180
lon_EP_min = 160
# Generate Niño 3.4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = atm_monthly_ctrl[ctrl_keys[0]].TS.isel(member=0, time=0)*0
nino4_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                1,zeros_mask)
nino4_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                1,zeros_mask)

nino4_mask = nino4_WP_mask + nino4_EP_mask
# Add cyclical point for ML 
nino4_mask_wrap, lon_wrap = add_cyclic_point(nino4_mask,coord=nino4_mask.lon)


# NOTE: This next step takes a few minutes to run. Can decrease runtime by changing atm_varnames_monthly_subset to be only variables that you need significance plotted
# atm_varnames_monthly_subset = ['TS']

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


# Calculate standard error of control ensemble for MCB deployment window
atm_mcb_on_sig = {}
for key in mcb_keys:
    atm_mcb_on_sig[key] = {}
    for varname in atm_varnames_monthly_subset:
        print(varname)
        # Subset MCB window
        mcb_on_start_dict = {'':2,'06-02':1,'06-08':1,'06-11':1,'09-02':4,'09-11':4,'12-02':7}
        mcb_on_end_dict = {'':5,'06-02':10,'06-08':4,'06-11':7,'09-02':10,'09-11':7,'12-02':10}
        tslice=atm_monthly_ctrl[ctrl_keys[0]][varname].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key]))
        tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
        tslice = tslice.rename({'time':'month'})
        tslice = fun.weighted_temporal_mean_clim(tslice)
        sem = stats.sem(tslice.values,axis=0)
        # Subset MCB anomaly dataarray for DJF of first year
        tslice2=atm_monthly_ensemble_anom[key][varname].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key]))
        tslice2 =tslice2.assign_coords(time=pd.to_datetime(tslice2.time.values).month)
        tslice2 = tslice2.rename({'time':'month'})
        tslice2 = fun.weighted_temporal_mean_clim(tslice2)
        atm_mcb_on_sig[key][varname] = xr.where(np.abs(tslice2)>2*np.abs(sem), 0,1)



## FIGURE 2b,c. MAPS OF SURFACE TEMPERATURE FOR CONTROL AND MCB RELATIVE TO HISTORICAL CLIMATOLOGY
subplot_lab = ['b','c']
fig = plt.figure(figsize=(4,5));
cmin=-3
cmax=3

# CONTROL
subplot_num = 0
t1=ts_ctrl_anom[''].isel(time=slice(4,16)).mean(dim='member')
ci_in='none'
ci_level='none'
ci_display='none'
# Subset DJF and rename by month
tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
tlabel='DJF '+str(pd.to_datetime(tslice.time.values).year[0]) + '-' +str(pd.to_datetime(tslice.time.values).year[-1])
tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
tslice = tslice.rename({'time':'month'})
# Calculate weighted temporal mean and assign units
in_xr = fun.weighted_temporal_mean_clim(tslice)
in_xr.attrs['units'] = '\N{DEGREE SIGN}C'
# Get mean value in seeding region for plot
mcb_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
nino34_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
summary_stat = [mcb_mean_val, np.nan]
axs, p = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                        CI_in=ci_in,CI_level=ci_level,CI_display=ci_display,\
                        projection='Robinson',nrow=2,ncol=1,subplot_num=subplot_num,mean_val='none',cbar=False)
plt.contour(lon_wrap,nino34_mask.lat,nino34_mask_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1.5,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title(subplot_lab[subplot_num],fontsize=14, fontweight='bold',loc='left');
# Add experiment as annotation
plt.annotate('DJF ' + year_init+'-'+str(int(year_init)+1)+' El Niño', xy=(.21,1.07), xycoords='axes fraction',color='k');
subplot_num+=1

# MCB 06-02
t1=atm_monthly_ensemble_anom['06-02']['TS'].isel(time=slice(4,16))
ci_in = atm_djf_sig['06-02']['TS']
ci_level=0.05
ci_display='inv_stipple'
# Subset DJF and rename by month
tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
tlabel='DJF '+str(pd.to_datetime(tslice.time.values).year[0]) + '-' +str(pd.to_datetime(tslice.time.values).year[-1])
tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
tslice = tslice.rename({'time':'month'})
# Calculate weighted temporal mean and assign units
in_xr = fun.weighted_temporal_mean_clim(tslice)
in_xr.attrs['units'] = '\N{DEGREE SIGN}C'
# Get mean value in seeding region for plot
mcb_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
nino34_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
summary_stat = [mcb_mean_val, np.nan]
fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                        CI_in=ci_in,CI_level=ci_level,CI_display=ci_display,\
                        projection='Robinson',nrow=2,ncol=1,subplot_num=subplot_num,mean_val='none',cbar=False)
plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.contour(lon_wrap,nino34_mask.lat,nino34_mask_wrap,\
        transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1.5,add_colorbar=False,\
        subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
plt.title(subplot_lab[subplot_num],fontsize=14, fontweight='bold',loc='left');
# Add experiment as annotation
plt.annotate('Full effort', xy=(.4,1.07), xycoords='axes fraction',color='k');
cbar_ax = fig.add_axes([0.115, 0.07, 0.8, 0.04]) #rect kwargs [left, bottom, width, height];
plt.colorbar(p, cax = cbar_ax, orientation='horizontal', label='Temperature (\N{DEGREE SIGN}C)', extend='both',pad=0.1);


## FIG ED4 (05-2015 opt)
## TS anom maps for DJF
plot_labels = ['a','b','c','d','e','f']
if sensitivity_opt=='n':
    fig = plt.figure(figsize=(8,5));
    subplot_num = 0
    ## Calculate the DJF mean for the first simulated year of the simulation
    # Subset first year of simulation
    t1=atm_monthly_ensemble_anom['']['TS'].isel(time=slice(4,16))
    ci_in = atm_djf_sig['']['TS']
    ci_level=0.05
    ci_display='inv_stipple'
    cmin=-2
    cmax=2
    # Subset DJF and rename by month
    tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    tlabel='DJF '+str(pd.to_datetime(tslice.time.values).year[0]) + '-' +str(pd.to_datetime(tslice.time.values).year[-1])
    tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
    tslice = tslice.rename({'time':'month'})
    # Calculate weighted temporal mean and assign units
    in_xr = fun.weighted_temporal_mean_clim(tslice)
    in_xr.attrs['units'] = '\N{DEGREE SIGN}C'
    # Get mean value in seeding region for plot
    mcb_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
    nino34_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
    summary_stat = [mcb_mean_val, np.nan]
    fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                            CI_in=ci_in,CI_level=ci_level,CI_display=ci_display,\
                            projection='Robinson',nrow=1,ncol=1,subplot_num=subplot_num,mean_val=summary_stat)
    plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.contour(lon_wrap,nino34_mask.lat,nino34_mask_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.title(tlabel+' Surface Temperature',fontsize=12, fontweight='bold',loc='left');
    # Add Nino3.4 mean value as annotation
    plt.annotate(str(round(nino34_mean_val,2))+ ' '+str(in_xr.units), xy=(.72,.85), xycoords='figure fraction',color='magenta');
    ## Save figure and close
    fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);
elif sensitivity_opt=='y':
    fig = plt.figure(figsize=(12,4));
    subplot_num = 0
    for key in mcb_keys:
        ## Calculate the DJF mean for the first simulated year of the simulation
        # Subset first year of simulation
        t1=atm_monthly_ensemble_anom[key]['TS'].isel(time=slice(4,16))
        ci_in=atm_djf_sig[key]['TS']        
        ci_level=0.05
        ci_display='inv_stipple'
        cmin=-2
        cmax=2
        # Subset DJF and rename by month
        tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
        tlabel='DJF '+str(pd.to_datetime(tslice.time.values).year[0]) + '-' +str(pd.to_datetime(tslice.time.values).year[-1])
        tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
        tslice = tslice.rename({'time':'month'})
        # Calculate weighted temporal mean and assign units
        in_xr = fun.weighted_temporal_mean_clim(tslice)
        in_xr.attrs['units'] = '\N{DEGREE SIGN}C'
        # Get mean value in seeding region for plot
        mcb_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
        nino34_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
        summary_stat = [mcb_mean_val, np.nan]
        t, p = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                                CI_in=ci_in,CI_level=ci_level,CI_display=ci_display,\
                                projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val=summary_stat,cbar=False)
        plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.contour(lon_wrap,nino34_mask.lat,nino34_mask_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
        plt.annotate(str(round(nino34_mean_val,2))+ ' '+str(in_xr.units), xy=(.84,.94), xycoords='axes fraction',color='magenta');
        # Add Nino3.4 mean value as annotation
        plt.annotate('MCB on: '+key, xy=(.32,1.05), xycoords='axes fraction',color='k');
        subplot_num += 1
    ## Save figure and close
    fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.1,hspace=0.15);
    cbar_ax = fig.add_axes([0.27, 0.04, 0.5, 0.04]) #rect kwargs [left, bottom, width, height];
    plt.colorbar(p, cax = cbar_ax, orientation='horizontal', label='Temperature (\N{DEGREE SIGN}C)', extend='both',pad=0.2); 


## MCB start/end TS sensitivity scatter plot
mcb_on_start_dict = {'06-02':1,'06-08':1,'06-11':1,'09-02':4,'09-11':4,'12-02':7}
mcb_on_end_dict = {'06-02':10,'06-08':4,'06-11':7,'09-02':10,'09-11':7,'12-02':10}
mcb_on_start_date_dict = {}
mcb_on_end_date_dict = {}
mcb_mean_val = {}
nino34_mean_val = {}
global_mean_val = {}
for key in mcb_keys:
    # Create list of start and dates
    mcb_on_start_date_dict[key]=atm_monthly_ensemble_anom[key]['TS'].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key])).time[0].values
    mcb_on_end_date_dict[key]=atm_monthly_ensemble_anom[key]['TS'].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key])).time[-1].values
    # Subset first year of simulation
    t1=atm_monthly_ensemble_anom[key]['TS'].isel(time=slice(4,16))
    # Subset DJF and rename by month
    tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    tlabel='DJF '+str(pd.to_datetime(tslice.time.values).year[0]) + '-' +str(pd.to_datetime(tslice.time.values).year[-1])
    tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
    tslice = tslice.rename({'time':'month'})
    # Calculate weighted temporal mean and assign units
    in_xr = fun.weighted_temporal_mean_clim(tslice)
    in_xr.attrs['units'] = '\N{DEGREE SIGN}C'
    # Create MCB mean values
    mcb_mean_val[key] = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
    # Create Nino3.4 mean values
    nino34_mean_val[key] = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
    # Create global mean values
    global_mean_val[key] = float(fun.calc_weighted_mean_tseries(in_xr))


## Contour plots
# Define MCB duration and months initialized prior to peak ENSO
mcb_duration = {'06-02':9,'06-08':3,'06-11':6,'09-02':6,'09-11':3,'12-02':3}
mcb_init = {'06-02':6,'06-08':6,'06-11':6,'09-02':3,'09-11':3,'12-02':0}


# Define percent change from Control
# Subset first year of simulation
t1=ts_ctrl_anom[''].isel(time=slice(4,16)).mean(dim='member')
# Subset DJF and rename by month
tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
tslice = tslice.rename({'time':'month'})
tslice = fun.weighted_temporal_mean_clim(tslice)
ctrl_mcb_mean_val = float(fun.calc_weighted_mean_tseries(tslice.where(seeding_mask_seed>0,drop=True)).values)
ctrl_nino34_mean_val = float(fun.calc_weighted_mean_tseries(tslice.where(nino34_mask>0,drop=True)).values)
ctrl_global_mean_val = float(fun.calc_weighted_mean_tseries(tslice))
mcb_percent_change = {}
nino34_percent_change = {}
global_percent_change = {}
for key in mcb_keys:
    print(key)
    # Create MCB mean values
    mcb_percent_change[key] = -mcb_mean_val[key]/(ctrl_mcb_mean_val-0.5)*100
    # Create Nino3.4 mean values
    nino34_percent_change[key] = -nino34_mean_val[key]/(ctrl_nino34_mean_val-0.5)*100
    # Create global mean values
    global_percent_change[key] = -global_mean_val[key]/(ctrl_global_mean_val-0.5)*100


## FIG ED1 (05-2015 and 05-1997; manually pasted together)
# Contour plot: Percent change from control
# need to output each El Niño year and a colorbar separately (3 parts)
if year_init=='2015':
    cmin=-100
    cmax=100
    levels = np.linspace(0,120,13)
    ccmap='Blues'
    plt.subplots(1,2,sharey=True,figsize=(8,4));
    # MCB region
    plt.subplot(1,2,1);
    plt.tricontourf(list(mcb_duration.values()), list(mcb_init.values()), list(mcb_percent_change.values()), cmap=ccmap, levels=levels,extend='both');
    plt.title('MCB region', fontsize=12);
    plt.ylabel('MCB start (months before DJF)',fontsize=12);
    plt.text(2, 6.5, 'a', size=14, weight='bold');
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        labelbottom=False) # labels along the bottom edge are off
    # plt.xlabel('MCB duration (months)',fontsize=12);
    #plt.colorbar(label='SST (% reduction)');

    # Nino3.4
    plt.subplot(1,2,2);
    plt.tricontourf(list(mcb_duration.values()), list(mcb_init.values()), list(nino34_percent_change.values()), cmap=ccmap, levels=levels,extend='both');
    plt.title('Nino3.4 region', fontsize=12);
    plt.text(2, 6.5, 'b', size=14, weight='bold');
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        labelbottom=False) # labels along the bottom edge are off#plt.xlabel('MCB duration (months)',fontsize=12);
    #plt.xlabel('MCB duration (months)',fontsize=12);
    #plt.colorbar(label='SST (% reduction)');
    # Plot aesthetics
    plt.tight_layout();
elif year_init=='1997':
    cmin=-100
    cmax=100
    levels = np.linspace(0,120,13)
    ccmap='Blues'
    plt.subplots(1,2,sharey=True,figsize=(8,4));
    # MCB region
    plt.subplot(1,2,1);
    plt.tricontourf(list(mcb_duration.values()), list(mcb_init.values()), list(mcb_percent_change.values()), cmap=ccmap, levels=levels,extend='both');
    plt.title('MCB region', fontsize=12);
    plt.ylabel('MCB start (months before DJF)',fontsize=12);
    plt.text(2, 6.5, 'c', size=14, weight='bold');
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=True,      # ticks along the bottom edge are off
    #     labelbottom=False) # labels along the bottom edge are off
    plt.xlabel('MCB duration (months)',fontsize=12);
    #plt.colorbar(label='SST (% reduction)');

    # Nino3.4
    plt.subplot(1,2,2);
    plt.tricontourf(list(mcb_duration.values()), list(mcb_init.values()), list(nino34_percent_change.values()), cmap=ccmap, levels=levels,extend='both');
    plt.title('Nino3.4 region', fontsize=12);
    plt.text(2, 6.5, 'd', size=14, weight='bold');
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=True,      # ticks along the bottom edge are off
    #     labelbottom=False) # labels along the bottom edge are off#plt.xlabel('MCB duration (months)',fontsize=12);
    plt.xlabel('MCB duration (months)',fontsize=12);
    # plt.colorbar(label='SST (% reduction)');
    # Plot aesthetics
    plt.tight_layout();



## FIG 2a (05-2015) and FIG ED2a (05-1997)
## Mask values and average over space for time series
# ERROR BARS
# Control
ts_ctrl_anom_sem_lower_plot = {}
ts_ctrl_anom_sem_upper_plot = {}
ts_ts_ctrl_anom_plot = {}
for key in ctrl_keys:
    print(key)
    # Each ensemble member
    ts_ctrl_anom_sem_lower_plot[key] = (ts_ctrl_anom[key].mean(dim='member') - ts_ctrl_anom_sem[key]).where(nino34_mask>0,drop=True).mean(dim=('lat','lon')).values
    ts_ctrl_anom_sem_upper_plot[key] = (ts_ctrl_anom[key].mean(dim='member') + ts_ctrl_anom_sem[key]).where(nino34_mask>0,drop=True).mean(dim=('lat','lon')).values
    # Ensemble mean
    ts_ts_ctrl_anom_plot[key] = ts_ctrl_anom[key].mean(dim='member').where(nino34_mask>0,drop=True).mean(dim=('lat','lon')).values
# MCB
ts_mcb_anom_sem_lower_plot = {}
ts_mcb_anom_sem_upper_plot = {}
ts_ts_mcb_anom_plot = {}
for key in mcb_keys:
    print(key)
    # Each ensemble member
    ts_mcb_anom_sem_lower_plot[key] = (ts_mcb_anom[key].mean(dim='member') - ts_mcb_anom_sem[key]).where(nino34_mask>0,drop=True).mean(dim=('lat','lon')).values
    ts_mcb_anom_sem_upper_plot[key] = (ts_mcb_anom[key].mean(dim='member') + ts_mcb_anom_sem[key]).where(nino34_mask>0,drop=True).mean(dim=('lat','lon')).values
    # Ensemble mean
    ts_ts_mcb_anom_plot[key] = ts_mcb_anom[key].mean(dim='member').where(nino34_mask>0,drop=True).mean(dim=('lat','lon')).values


# Colormaps created w/ colorbrewer (https://colorbrewer2.org/#type=qualitative&scheme=Accent&n=7)
mcb_colors = {'':'#a50f15','06-02':'#a50f15','06-08':'#a50f15','06-11':'#a50f15','09-02':'#ef3b2c','09-11':'#ef3b2c','12-02':'#fc9272'} # reds=start month
mcb_linestyle = {'':'solid','06-02':'solid','06-08':(0, (1, 1)),'06-11':'dashed','09-02':'dashed','09-11':(0, (1, 1)),'12-02':(0, (1, 1))} # linestyle=duration
# Subset MCB window
mcb_on_start_dict = {'06-02':1,'06-08':1,'06-11':1,'09-02':4,'09-11':4,'12-02':7}
mcb_on_end_dict = {'06-02':10,'06-08':4,'06-11':7,'09-02':10,'09-11':7,'12-02':10}

# Set figure dimensions and grid
fig = plt.figure(figsize=(6, 5),layout='constrained')
spec = fig.add_gridspec(3, 1)

### MAIN TIME SERIES SUBPLOT
ax0 = fig.add_subplot(spec[0:2, 0])
# Control
for key in ctrl_keys:
    # PLOT 2 STANDARD ERRORS
    plt.fill_between(ts_ctrl_anom_sem[key].time, ts_ctrl_anom_sem_lower_plot[key], ts_ctrl_anom_sem_upper_plot[key],color='k', alpha=0.2)
    # PLOT ENSEMBLE MEAN
    plt.plot(ts_ctrl_anom[key].time,ts_ts_ctrl_anom_plot[key],color='k',linewidth=3,label='Control');
# MCB
for key in mcb_keys:
    # PLOT ENSEMBLE MEAN
    plt.plot(ts_mcb_anom[key].time,ts_ts_mcb_anom_plot[key],color=mcb_colors[key],linestyle=mcb_linestyle[key],linewidth=3,label='MCB '+key);
# PLOT PEAK ENSO DJF
# Subset first year of simulation
t1=ts_ctrl_anom[''].isel(time=slice(4,16))
# Subset DJF and rename by month
tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
plt.fill_between(tslice.time, -5, 5,color='steelblue', alpha=0.2);
# PLOT AESTHETICS
plt.ylim(-3.5,3.5);
if enso_phase=='nino':
    plt.axhline(0.5, linestyle='--', color='k');
    plt.text(tslice.time[1], -3, s='ENSO peak', rotation=90, fontsize=12, color='steelblue');
elif enso_phase=='nina':
    plt.axhline(-0.5, linestyle='--', color='k');
    plt.text(tslice.time[1], 1, s='ENSO peak', rotation=90, fontsize=12, color='steelblue');
# Format dates
ax=plt.gca();
xbounds=ax.get_xlim();
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
plt.xlabel('Time',fontsize=12); plt.ylabel('Niño 3.4 SST anomaly (\N{DEGREE SIGN}C)', fontsize=12);
plt.title('a', fontsize=14, fontweight='bold',loc='left');

### Add legend 
ax1 = fig.add_subplot(spec[2, 0])
mcb_legend_y = {'06-02':6,'06-08':4,'06-11':5,'09-02':3,'09-11':2,'12-02':1}
mcb_legend_label = {'06-02':'Jun-Feb','06-08':'Jun-Aug','06-11':'Jun-Nov','09-02':'Sep-Feb','09-11':'Sep-Nov','12-02':'Dec-Feb'}
mcb_legend_longname = {'06-02':'Full effort','06-08':'Early action','06-11':'','09-02':'','09-11':'','12-02':'11th hour'}
ax1.set_xlim(xbounds);
ax1.set_ylim(0.5,6.5);
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
plt.ylabel('MCB scenario', fontsize=12);
ax1.set_xticks([]);ax1.set_yticks([]);
plt.setp(ax1.spines.values(), color=None);


## FIG ED3 (05-2015)
## SWCF anom maps for MCB period/Jun-Feb
if sensitivity_opt=='n':
    fig = plt.figure(figsize=(8,5));
    subplot_num = 0
    cmin=-40
    cmax=40
    ## Calculate the MCB mean for the first simulated year of the simulation
    # Subset MCB period
    tslice=atm_monthly_ensemble_anom['']['SWCF'].isel(time=slice(2,5))
    tlabel='MCB window'
    tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
    tslice = tslice.rename({'time':'month'})
    # Calculate weighted temporal mean and assign units
    in_xr = fun.weighted_temporal_mean_clim(tslice)
    in_xr.attrs['units'] = 'W/m$^{2}$'
    # Get mean value in seeding region for plot
    mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
    summary_stat = [mean_val, np.nan]
    fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='bwr', plot_zoom='global', central_lon=180,\
                            CI_in=atm_mcb_on_sig['']['SWCF'],CI_level=0.05,CI_display='inv_stipple',\
                            projection='Robinson',nrow=1,ncol=1,subplot_num=subplot_num,mean_val=summary_stat)
    plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.title('3-month ' + tlabel+' SWCF',fontsize=12, fontweight='bold',loc='left');
    ## Save figure and close
    fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);
elif sensitivity_opt=='y':
    fig = plt.figure(figsize=(12,4));
    subplot_num = 0
    cmin=-40
    cmax=40
    for key in mcb_keys:
        # Calculate SWCF for Jun to Feb
        if month_init=='05':
            tslice=atm_monthly_ensemble_anom[key]['SWCF'].isel(time=slice(1,10))
        # Subset DJF and rename by month
        tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
        tslice = tslice.rename({'time':'month'})
        # Calculate weighted temporal mean and assign units
        in_xr = fun.weighted_temporal_mean_clim(tslice)
        in_xr.attrs['units'] = 'W/m$^{2}$'
        # Get mean value in seeding region for plot
        mcb_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
        nino34_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
        if abs(mcb_mean_val)>10:
            summary_stat = [round(mcb_mean_val,1), np.nan] 
        elif abs(mcb_mean_val)<10:
            summary_stat = [round(mcb_mean_val,2), np.nan] 
        t, p = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='bwr', plot_zoom='global', central_lon=180,\
                                CI_in=atm_mcb_on_sig[key]['SWCF'],CI_level=0.05,CI_display='inv_stipple',\
                                projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val=summary_stat,cbar=False)
        plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.contour(lon_wrap,nino34_mask.lat,nino34_mask_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        # Add Nino3.4 mean value as annotation
        if abs(nino34_mean_val)>10:
            plt.annotate(str(round(nino34_mean_val,1))+ ' '+str(in_xr.units), xy=(.84,.94), xycoords='axes fraction',color='magenta');
        elif abs(nino34_mean_val)<10:
            plt.annotate(str(round(nino34_mean_val,2))+ ' '+str(in_xr.units), xy=(.84,.94), xycoords='axes fraction',color='magenta');
        plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
        # Add Nino3.4 mean value as annotation
        plt.annotate('MCB on: '+key, xy=(.32,1.05), xycoords='axes fraction',color='k');
        subplot_num += 1
    ## Save figure and close
    fig.subplots_adjust(bottom=0.1, top=0.9, wspace=0.1,hspace=0.15);
    cbar_ax = fig.add_axes([0.27, 0.04, 0.5, 0.04]) #rect kwargs [left, bottom, width, height];
    plt.colorbar(p, cax = cbar_ax, orientation='horizontal', label='SW radiative forcing (W/m$^{2}$)', extend='both',pad=0.2); 
