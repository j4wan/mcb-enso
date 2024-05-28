### PURPOSE: Script to compare SMYLE AUFIRE and SMYLE MCB
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/28/2024

### NOTES: adapted from smyle_aus_mcb_comparison_v1.py

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

## READ IN DATA
# Define simulation keys
sim_keys = ['aus','nina']
# Define output variables needed for each experiment as dictionary
atm_monthly_ctrl_clim_xr = {}
ts_clim_ensemble_mean = {}
atm_monthly_ctrl = {}
ts_ctrl_anom={}
ts_ctrl_anom_std={}
ts_ctrl_anom_sem={}
atm_monthly_mcb={}
ts_mcb_anom={}
ts_mcb_anom_std={}
ts_mcb_anom_sem={}
atm_monthly_anom = {}
atm_monthly_ensemble_anom = {}

# Loop through each experiment to read and process data
for sim in sim_keys:
    if sim=='nina':
        ## 2019-2020 LA NIÑA + MCB
        month_init = '11'
        year_init = '2019'
        enso_phase = 'nina'
        sensitivity_opt = 'n'
    elif sim=='nino':
        ## 2015-2016 EL NIÑO + MCB
        month_init = '11'
        year_init = '2015'
        enso_phase = 'nino'
        sensitivity_opt = 'n'
    elif sim=='aus':
        ## 2019-2020 LA NIÑA + AUFIRE
        month_init = '08'
        year_init = '2019'
        enso_phase = 'nina'
        sensitivity_opt = 'n'
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
    if( sim=='nino') or (sim=='nina'):
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
    elif sim=='aus':
        mcb_keys=['']
        for key in mcb_keys:
            mcb_files = []
            for yr in yr_init:    
                mcb_files = mcb_files + glob.glob('/_data/SMYLE-MCB/SMYLE-AUFIRE/b.e21.BSMYLE-AUFIRE.f09_g17.'+yr+'*-'+month_init+'.*')
            mcb_members = []
            for i in mcb_files:
                start = i.find('f09_g17.') + len('f09_g17.')
                tmp = i[start:None]
                if tmp not in mcb_members:
                    mcb_members.append(tmp)
            mcb_members = sorted(mcb_members)
            print(mcb_members)
            mcb_sims[key] = mcb_members[:len(ctrl_members)] #only take first 20 members to match control



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
    atm_varnames_monthly_subset = ['SWCF','TS','PRECT','U','V']

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
    if len(os.listdir(target_dir))==0:
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
        atm_monthly_ctrl_clim_xr[sim] = xr.concat(list(map(atm_monthly_ctrl_clim.get, clim_members)),pd.Index(clim_members,name='member'))
        ## Convert time to datetime index
        atm_monthly_ctrl_clim_xr[sim] = atm_monthly_ctrl_clim_xr[sim].assign_coords(time=atm_monthly_ctrl_clim_xr[sim].indexes['time'].to_datetimeindex())
        ## Convert units
        # PRECT
        m_to_mm = 1e3 #mm/m
        s_to_days = 86400 #s/day
        # Convert from m/s to mm/day
        atm_monthly_ctrl_clim_xr[sim] = atm_monthly_ctrl_clim_xr[sim].assign(PRECT=atm_monthly_ctrl_clim_xr[sim]['PRECT']*m_to_mm*s_to_days)
        atm_monthly_ctrl_clim_xr[sim]['PRECT'].attrs['units'] = 'mm/day'
        # TS
        # Convert from K to C
        atm_monthly_ctrl_clim_xr[sim] = atm_monthly_ctrl_clim_xr[sim].assign(TS=atm_monthly_ctrl_clim_xr[sim]['TS']-273.15)
        atm_monthly_ctrl_clim_xr[sim]['TS'].attrs['units'] = '°C'
        ### EXPORT PROCESSED NETCDF
        atm_monthly_ctrl_clim_xr[sim].to_netcdf(target_dir+'/BSMYLE.'+str(pd.to_datetime(atm_monthly_ctrl_clim_xr[sim].time.values[0]).year)+'-'+str(pd.to_datetime(atm_monthly_ctrl_clim_xr[sim].time.values[-1]).year)+'-'+month_init+'.atm_tseries_combined.nc',mode='w',format='NETCDF4')
    else:
        atm_monthly_ctrl_clim_xr[sim] = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE-MCB/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*atm_tseries_combined.nc')[0])))

    # Compute climatogical mean from 1970-2014
    tslice = atm_monthly_ctrl_clim_xr[sim].loc[{'time':[t for t in pd.to_datetime(atm_monthly_ctrl_clim_xr[sim].time.values) if (t.year<2015)]}]
    # ts_clim_ensemble_mean[sim] = tslice.mean(dim=('member','time')) # By annual climatology
    ts_clim_ensemble_mean[sim] = tslice.TS.mean(dim=('member')).groupby('time.month').mean() # By monthly climatology



    ## READ IN CONTROL SIMULATION & PRE-PROCESS
    # ATM
    atm_monthly_ctrl[sim]={}
    ts_ctrl_anom[sim]={}
    ts_ctrl_anom_std[sim]={}
    ts_ctrl_anom_sem[sim]={}
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
        atm_monthly_ctrl[sim][key] = xr.concat(list(map(atm_monthly_ctrl_single_mem.get, intersect_members)),pd.Index(intersect_members,name='member'))
        # Convert time to datetime index
        atm_monthly_ctrl[sim][key] = atm_monthly_ctrl[sim][key].assign_coords(time=atm_monthly_ctrl[sim][key].indexes['time'].to_datetimeindex())
        ## PRECT
        # # Convert from m/s to mm/day
        m_to_mm = 1e3 #mm/m
        s_to_days = 86400 #s/day
        atm_monthly_ctrl[sim][key] = atm_monthly_ctrl[sim][key].assign(PRECT=atm_monthly_ctrl[sim][key]['PRECT']*m_to_mm*s_to_days)
        atm_monthly_ctrl[sim][key]['PRECT'].attrs['units'] = 'mm/day'
        ## TS
        # Convert from K to C
        atm_monthly_ctrl[sim][key] = atm_monthly_ctrl[sim][key].assign(TS=atm_monthly_ctrl[sim][key]['TS']-273.15)
        atm_monthly_ctrl[sim][key]['TS'].attrs['units'] = '°C'
        ##DRIFT CORRECTION
        # Compute drift correction anomaly
        # # By annual climatology
        # ts_ctrl_anom[sim][key]=atm_monthly_ctrl[sim][key]['TS']-ts_clim_ensemble_mean[sim]
        # By month climatology
        i_month=np.arange(1,13,1)
        ts_ctrl_anom[sim][key] = atm_monthly_ctrl[sim][key]['TS']*1
        for month in i_month:
            ts_ctrl_anom[sim][key].loc[{'time':[t for t in pd.to_datetime(ts_ctrl_anom[sim][key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean[sim].sel(month=month)
        ts_ctrl_anom[sim][key].attrs['units']='\N{DEGREE SIGN}C'
        # Compute standard deviation
        ts_ctrl_anom_std[sim][key]=ts_ctrl_anom[sim][key].std(dim='member')
        # Compute twice standard error
        ts_ctrl_anom_sem[sim][key]=2 * ts_ctrl_anom[sim][key].std(dim='member')/np.sqrt(len(ts_ctrl_anom[sim][key].member))



    ## READ IN MCB SIMULATIONS & PRE-PROCESS
    # ATM
    atm_monthly_mcb[sim]={}
    ts_mcb_anom[sim]={}
    ts_mcb_anom_std[sim]={}
    ts_mcb_anom_sem[sim]={}
    for key in mcb_keys:
        atm_monthly_mcb_single_mem = {}
        for m in mcb_sims[key]:
            print(m)
            if (sim=='nino') or (sim=='nina'):
                dir_mcb = glob.glob('/_data/SMYLE-MCB/MCB/b.e21.BSMYLE.f09_g17.MCB*'+m+'/atm/proc/tseries/month_1')[0]
            elif sim=='aus':
                dir_mcb = glob.glob('/_data/SMYLE-MCB/SMYLE-AUFIRE/b.e21.BSMYLE-AUFIRE.f09_g17.*'+m+'/atm/proc/tseries/month_1')[0]
            file_subset_ctrl = []
            file_subset_mcb = []
            for var in atm_varnames_monthly_subset:
                pattern = "."+var+"."
                var_file_mcb = [f for f in os.listdir(dir_mcb) if pattern in f]
                file_subset_mcb.append(dir_mcb+'/'+var_file_mcb[0])
            atm_monthly_mcb_single_mem[m] = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_mfdataset(file_subset_mcb)))
        # Combine all files into one xarray dataset with ensemble members as a new dimension
        atm_monthly_mcb[sim][key] = xr.concat(list(map(atm_monthly_mcb_single_mem.get, mcb_sims[key])),pd.Index(intersect_members,name='member'))
        # Convert time to datetime index
        atm_monthly_mcb[sim][key] = atm_monthly_mcb[sim][key].assign_coords(time=atm_monthly_mcb[sim][key].indexes['time'].to_datetimeindex())
        # Overwrite lat, lon to match control to fix rounding errors
        atm_monthly_mcb[sim][key] = atm_monthly_mcb[sim][key].assign_coords(lat= atm_monthly_ctrl[sim][ctrl_keys[0]].lat, lon= atm_monthly_ctrl[sim][ctrl_keys[0]].lon)
        ## PRECT
        # # Convert from m/s to mm/day
        m_to_mm = 1e3 #mm/m
        s_to_days = 86400 #s/day
        atm_monthly_mcb[sim][key] = atm_monthly_mcb[sim][key].assign(PRECT=atm_monthly_mcb[sim][key]['PRECT']*m_to_mm*s_to_days)
        atm_monthly_mcb[sim][key]['PRECT'].attrs['units'] = 'mm/day'
        ## TS
        # Convert from K to C
        atm_monthly_mcb[sim][key] = atm_monthly_mcb[sim][key].assign(TS=atm_monthly_mcb[sim][key]['TS']-273.15)
        atm_monthly_mcb[sim][key]['TS'].attrs['units'] = '°C'
        ##DRIFT CORRECTION
        # Compute drift correction anomaly
        # # By annual climatology
        # ts_mcb_anom[sim][key]=atm_monthly_mcb[sim][key]['TS']-ts_clim_ensemble_mean[sim]
        # By month climatology
        i_month=np.arange(1,13,1)
        ts_mcb_anom[sim][key] = atm_monthly_mcb[sim][key]['TS']*1
        for month in i_month:
            ts_mcb_anom[sim][key].loc[{'time':[t for t in pd.to_datetime(ts_mcb_anom[sim][key].time.values) if t.month==month]}]-=ts_clim_ensemble_mean[sim].sel(month=month)
        ts_mcb_anom[sim][key].attrs['units']='\N{DEGREE SIGN}C'
        # Compute standard deviation
        ts_mcb_anom_std[sim][key]=ts_mcb_anom[sim][key].std(dim='member')
        # Compute twice standard error
        ts_mcb_anom_sem[sim][key]=2 * ts_mcb_anom[sim][key].std(dim='member')/np.sqrt(len(ts_mcb_anom[sim][key].member))


    #%% COMPUTE ANOMALIES FOR SELECT VARIABLES
    ## 1a) MONTHLY ATMOSPHERE
    # Create empty dictionaries for anomalies
    atm_monthly_anom[sim] = {}
    atm_monthly_ensemble_anom[sim] = {}

    ## Loop through subsetted varnames list. 
    print('##ATM MONTHLY##')
    for key in mcb_keys:
        print(key)
        atm_monthly_anom[sim][key] = {}
        atm_monthly_ensemble_anom[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            atm_monthly_anom[sim][key][varname] = atm_monthly_mcb[sim][key][varname] - atm_monthly_ctrl[sim][ctrl_keys[0]][varname]
            atm_monthly_anom[sim][key][varname].attrs['units'] = atm_monthly_ctrl[sim][ctrl_keys[0]][varname].units
            atm_monthly_ensemble_anom[sim][key][varname] = atm_monthly_anom[sim][key][varname].mean(dim='member')
            atm_monthly_ensemble_anom[sim][key][varname].attrs['units'] = atm_monthly_ctrl[sim][ctrl_keys[0]][varname].units



#%% MASK DATA AND CALCULATE SIGNIFICANCE
atm_monthly_sig = {}
atm_djf_sig = {}
atm_mcb_on_sig = {}

for sim in sim_keys:
    # Get overlay mask files (area is the same for all of them so can just pick one)
    seeding_mask = fun.reorient_netCDF(xr.open_dataset('/_data/sesp_mask_CESM2_0.9x1.25_v3.nc'))

    # Force seeding mask lat, lon to equal the output CESM2 data (rounding errors)
    seeding_mask = seeding_mask.assign_coords({'lat':atm_monthly_ctrl[sim][ctrl_keys[0]]['lat'], 'lon':atm_monthly_ctrl[sim][ctrl_keys[0]]['lon']})
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
    zeros_mask = atm_monthly_ctrl[sim][ctrl_keys[0]].TS.isel(member=0, time=0)*0
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
    zeros_mask = atm_monthly_ctrl[sim][ctrl_keys[0]].TS.isel(member=0, time=0)*0
    nino4_WP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                    (zeros_mask.lon>=lon_WP_min) & (zeros_mask.lon<=lon_WP_max),\
                                    1,zeros_mask)
    nino4_EP_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                    (zeros_mask.lon>=lon_EP_min) & (zeros_mask.lon<=lon_EP_max),\
                                    1,zeros_mask)

    nino4_mask = nino4_WP_mask + nino4_EP_mask
    # Add cyclical point for ML 
    nino4_mask_wrap, lon_wrap = add_cyclic_point(nino4_mask,coord=nino4_mask.lon)
            

    # Identify signficant cells (ensemble mean differences > 2*SE)
    # Calculate standard error of control ensemble
    atm_monthly_sig[sim] = {}
    for key in mcb_keys:
        atm_monthly_sig[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            sem = stats.sem(atm_monthly_ctrl[sim][ctrl_keys[0]][varname].values,axis=0)
            atm_monthly_sig[sim][key][varname] = xr.where(np.abs(atm_monthly_ensemble_anom[sim][key][varname])>2*np.abs(sem), 0,1)

    # Calculate standard error of control ensemble for DJF of ENSO event
    atm_djf_sig[sim] = {}
    for key in mcb_keys:
        atm_djf_sig[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            # Subset first year of simulation
            t1=atm_monthly_ctrl[sim][ctrl_keys[0]][varname].isel(time=slice(4,16))
            # Subset DJF and rename by month
            tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
            tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
            tslice = tslice.rename({'time':'month'})
            tslice = fun.weighted_temporal_mean_clim(tslice)
            sem = stats.sem(tslice.values,axis=0)
            # Subset MCB anomaly dataarray for DJF of first year
            t2=atm_monthly_ensemble_anom[sim][key][varname].isel(time=slice(4,16))
            tslice2 =t2.loc[{'time':[t for t in pd.to_datetime(t2.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
            tslice2 =tslice2.assign_coords(time=pd.to_datetime(tslice2.time.values).month)
            tslice2 = tslice2.rename({'time':'month'})
            tslice2 = fun.weighted_temporal_mean_clim(tslice2)
            atm_djf_sig[sim][key][varname] = xr.where(np.abs(tslice2)>2*np.abs(sem), 0,1)


    # Calculate standard error of control ensemble for MCB deployment window
    atm_mcb_on_sig[sim] = {}
    for key in mcb_keys:
        atm_mcb_on_sig[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            # Subset MCB window
            if (sim=='nino') or (sim=='nina'):
                mcb_on_start_dict = {'':2,'06-02':1,'06-08':1,'06-11':1,'09-02':4,'09-11':4,'12-02':7}
                mcb_on_end_dict = {'':5,'06-02':10,'06-08':4,'06-11':7,'09-02':10,'09-11':7,'12-02':10}
            elif sim=='aus':
                # mcb_on_start_dict = {'':8} # to match Feb init MCB
                # mcb_on_end_dict = {'':11}
                mcb_on_start_dict = {'':4} # to match Nov init MCB
                mcb_on_end_dict = {'':7}
            tslice=atm_monthly_ctrl[sim][ctrl_keys[0]][varname].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key]))
            tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
            tslice = tslice.rename({'time':'month'})
            tslice = fun.weighted_temporal_mean_clim(tslice)
            sem = stats.sem(tslice.values,axis=0)
            # Subset MCB anomaly dataarray for DJF of first year
            tslice2=atm_monthly_ensemble_anom[sim][key][varname].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key]))
            tslice2 =tslice2.assign_coords(time=pd.to_datetime(tslice2.time.values).month)
            tslice2 = tslice2.rename({'time':'month'})
            tslice2 = fun.weighted_temporal_mean_clim(tslice2)
            atm_mcb_on_sig[sim][key][varname] = xr.where(np.abs(tslice2)>2*np.abs(sem), 0,1)



## FIG. 1
#%% PLOT MCB WINDOW SWCF, DJF PRECT, AND DJF TS MAPS (3X3)
plot_labels = ['a','b','c','d','e','f']
fig = plt.figure(figsize=(12,4));
subplot_num = 0
for sim in sim_keys:
    # MCB WINDOW SWCF
    cmin=-40
    cmax=40
    ## Calculate the MCB mean for the first simulated year of the simulation
    # Subset MCB period
    if sim=='nina':
        tslice=atm_monthly_ensemble_anom[sim]['']['SWCF'].isel(time=slice(2,5))
    elif sim=='aus':
        # tslice=atm_monthly_ensemble_anom[sim]['']['SWCF'].isel(time=slice(8,11))
        tslice=atm_monthly_ensemble_anom[sim]['']['SWCF'].isel(time=slice(4,7))
    tlabel='MCB window'
    tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
    tslice = tslice.rename({'time':'month'})
    # Calculate weighted temporal mean and assign units
    in_xr = fun.weighted_temporal_mean_clim(tslice)
    in_xr.attrs['units'] = 'W/m2'
    # Get mean value in seeding region for plot
    mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
    summary_stat = [mean_val, np.nan]
    # print(sim, summary_stat) #print values for main text
    swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='bwr', plot_zoom='global', central_lon=180,\
                            CI_in=atm_mcb_on_sig[sim]['']['SWCF'],CI_level=0.05,CI_display='inv_stipple',\
                            projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=False)
    plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
    subplot_num+=1
    
    # DJF PRECT
    lev_sfc = float(atm_monthly_mcb[sim][key].lev[-1].values)
    cmin=-2
    cmax=2
    ## Calculate the DJF mean for the first simulated year of the simulation
    # Subset first year of simulation
    t1=atm_monthly_ensemble_anom[sim]['']['PRECT'].isel(time=slice(4,16))
    # Subset DJF and rename by month
    tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    tlabel='DJF '+str(pd.to_datetime(tslice.time.values).year[0]) + '-' +str(pd.to_datetime(tslice.time.values).year[-1])
    tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
    tslice = tslice.rename({'time':'month'})
    # Calculate weighted temporal mean and assign units
    in_xr = fun.weighted_temporal_mean_clim(tslice)
    in_xr.attrs['units'] = 'mm/day'
    # Get mean value in seeding region for plot
    mcb_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
    nino34_mean_val = float(fun.calc_weighted_mean_tseries(in_xr.where(nino34_mask>0,drop=True)).values)
    summary_stat = [mcb_mean_val, np.nan]
    prect, p2 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='BrBG', plot_zoom='global', central_lon=180,\
                            #CI_in=atm_djf_sig['PRECT'],CI_level=0.05,CI_display='inv_stipple',\
                            projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=False)
    m1 = plt.quiver(atm_monthly_mcb[sim][key]['U'].lon.values[::10], atm_monthly_mcb[sim][key]['U'].lat.values[::10],\
                atm_monthly_mcb[sim][key].mean(dim='member').isel(time=8).sel(lev=lev_sfc)['U'].values[::10,::10],atm_monthly_mcb[sim][key].mean(dim='member').isel(time=8).sel(lev=lev_sfc)['V'].values[::10,::10],\
                transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.0025);
    plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
    subplot_num+=1

    # DJF TS
    t1=atm_monthly_ensemble_anom[sim]['']['TS'].isel(time=slice(4,16))
    ci_in = atm_djf_sig[sim]['']['TS']
    ci_level=0.05
    ci_display='inv_stipple'
    cmin=-1.5
    cmax=1.5
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
    global_mean_val = float(fun.calc_weighted_mean_tseries(in_xr).values)
    # print(sim, global_mean_val) # print values for main text
    summary_stat = [mcb_mean_val, np.nan]
    ts, p3 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                            CI_in=ci_in,CI_level=ci_level,CI_display=ci_display,\
                            projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=False)
    plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='k', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.contour(lon_wrap,nino34_mask.lat,nino34_mask_wrap,\
            transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='magenta', linewidths=1.5,add_colorbar=False,\
            subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
    plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
    subplot_num+=1
fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);
## Add PRECT quiver key
plt.quiverkey(m1, X=6.5, Y=0.82, U= 10, label='10 ms$^{-1}$', labelpos='E', coordinates = 'inches');
## Add experiment labels to first column
plt.annotate('2020-2021 La Niña + wildfires', xy=(.085,1.02), xycoords='figure fraction',color='k');
plt.annotate('2020-2021 La Niña + MCB', xy=(.095,.57), xycoords='figure fraction',color='k');
## Add colorbars to bottom of figure
cbar_ax = fig.add_axes([0.12, 0.07, 0.25, 0.025]) #rect kwargs [left, bottom, width, height];
plt.colorbar(p1, cax = cbar_ax, orientation='horizontal', label='SW radiative forcing (W/m$^{2}$)', extend='both',pad=0.1);
cbar_ax = fig.add_axes([0.385, 0.07, 0.25, 0.025]) #rect kwargs [left, bottom, width, height];
plt.colorbar(p2, cax = cbar_ax, orientation='horizontal', label='Precipitation (mm/day)', extend='both',pad=0.1);
cbar_ax = fig.add_axes([0.655, 0.07, 0.25, 0.025]) #rect kwargs [left, bottom, width, height];
plt.colorbar(p3, cax = cbar_ax, orientation='horizontal', label='Temperature (\N{DEGREE SIGN}C)', extend='both',pad=0.1);