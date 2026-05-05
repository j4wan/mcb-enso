### PURPOSE: Script to compare SMYLE AUFIRE and SMYLE MCB
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/05/2026
### NOTES: adapted from smyle_aus_mcb_comparison_v5.py

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
import string
plt.ion(); #uncomment for interactive plotting

dask.config.set({"array.slicing.split_large_chunks": False})

##################################################################################################################

#%% DEFINE FUNCTIONS
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
    soi_mean = ((ps_tah-ps_dar)/ps_diff_std).mean(dim='member')
    soi_sem = 2*((ps_tah-ps_dar)/ps_diff_std).std(dim='member')/np.sqrt(len(in_xr.member))
    return soi_mean,soi_sem

##################################################################################################################

## READ IN DATA
# Define simulation keys
sim_keys = ['aus','mcb_100cc_AFCDNC']
# Is the MCB run the same initialization month as the AUFIRE ensemble? (Aug-init)
same_init='y'
sim_label = {'aus':'','mcb_500cc':'.','mcb_100cc_AFCDNC':'_2019-08_100cc_AFCDNC'}

# Define output variables needed for each experiment as dictionary
atm_monthly_ctrl_clim_xr = {}
atm_monthly_ctrl = {}
atm_monthly_mcb = {}
atm_monthly_anom = {}
atm_monthly_ensemble_anom = {}

ts_ctrl_drift = {}
ps_ctrl_drift = {}

ts_ctrl_anom = {}
ps_ctrl_anom = {}
ts_mcb_anom = {}
ps_mcb_anom = {}

atm_monthly_sig = {}
atm_djf_sig = {}
atm_mcb_on_sig = {}

soi_ctrl = {}
soi_ctrl_sem = {}
soi_mcb = {}
soi_mcb_sem = {}

nino34_ctrl = {}
nino34_ctrl_sem = {}
nino34_mcb = {}
nino34_mcb_sem = {}

nino4_ctrl = {}
nino4_ctrl_sem = {}
nino4_mcb = {}
nino4_mcb_sem = {}


# Loop through each experiment to read and process data
# n_ens = input('How many ensemble members (Control and AUFIRE) do you want to read in? (1) first 20, (2) first 10, (3) last 10, (4) full 30): ')
n_ens = '1'
for sim in sim_keys:
    print(sim)
    if (sim=='mcb_100cc') or (sim=='mcb_300cc') or (sim=='mcb_500cc') or (sim=='mcb_300cc_ST') or (sim=='mcb_300cc_ET') or (sim=='mcb_300cc_ETST') or ((same_init=='n') and (sim=='mcb_100cc_AFCDNC')):
        ## 2019-2020 LA NIÑA + MCB
        month_init = '11'
        year_init = '2019'
        enso_phase = 'nina'
    elif (sim=='aus') or ((same_init=='y') and (sim=='mcb_100cc_AFCDNC')):
        ## 2019-2020 LA NIÑA + AUFIRE
        month_init = '08'
        year_init = '2019'
        enso_phase = 'nina'
    # Get list of control ensemble members
    if year_init=='1997':
        yr_init = ['1996','1997']
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
    #print(ctrl_members) 
    print('Initialization month: ', month_init,'-',year_init)

    # Get list of MCB ensemble members
    mcb_sims = {}
    if sim=='aus':
        mcb_keys=['']
        for key in mcb_keys:
            mcb_files = []
            for yr in yr_init:    
                mcb_files = mcb_files + glob.glob('/_data/SMYLE-AUFIRE/b.e21.BSMYLE-AUFIRE.f09_g17.'+yr+'*-'+month_init+'.*')
            mcb_members = []
            for i in mcb_files:
                start = i.find('f09_g17.') + len('f09_g17.')
                tmp = i[start:None]
                if tmp not in mcb_members:
                    mcb_members.append(tmp)
            mcb_members = sorted(mcb_members)
            #print(mcb_members)
            if len(ctrl_members)==len(mcb_members):
                if n_ens=='1':
                    mcb_sims[key] = sorted(mcb_members)[:20] # first 20 members to match historical ensemble
                    intersect_members = sorted(ctrl_members)[:20]
                elif n_ens=='2':
                    mcb_sims[key] = sorted(mcb_members)[:10] #only take first 10 members
                    intersect_members = sorted(ctrl_members)[:10]
                elif n_ens=='3':
                    mcb_sims[key] = sorted(mcb_members)[-10:] #only take last 10 members
                    intersect_members = sorted(ctrl_members)[-10:]
                elif n_ens=='4':
                    mcb_sims[key] = sorted(mcb_members) #full ensemble (30 members)
                    intersect_members = sorted(ctrl_members)
                print('AUFIRE members:', mcb_sims[key])
                print('Control SMYLE members:',intersect_members)
            else:
                print('ERROR: Cannot find intersection beccause ensemble sizes are different')
    else:
        mcb_keys=['']
        for key in mcb_keys:
            mcb_files = []
            for yr in yr_init:    
                mcb_files = mcb_files + glob.glob('/_data/MCB/b.e21.BSMYLE.f09_g17.MCB'+sim_label[sim]+'*'+yr+'*-'+month_init+'.*')
            mcb_members = []
            for i in mcb_files:
                start = i.find('f09_g17.MCB') + len('f09_g17.MCB.')
                tmp = i[start:None]
                if tmp not in mcb_members:
                    mcb_members.append(tmp)
            mcb_members = sorted(mcb_members)
            #print(mcb_members)
            mcb_sims[key] = mcb_members
            if len(ctrl_members)==len(mcb_members):
                if n_ens=='1':
                    mcb_sims[key] = sorted(mcb_members)[:20] # first 20 members to match historical ensemble
                    intersect_members = sorted(ctrl_members)[:20]
                elif n_ens=='2':
                    mcb_sims[key] = sorted(mcb_members)[:10] #only take first 10 members
                    intersect_members = sorted(ctrl_members)[:10]
                elif n_ens=='3':
                    mcb_sims[key] = sorted(mcb_members)[-10:] #only take last 10 members
                    intersect_members = sorted(ctrl_members)[-10:]
                elif n_ens=='4':
                    mcb_sims[key] = sorted(mcb_members) #full ensemble (30 members)
                    intersect_members = sorted(ctrl_members)
                print('AUFIRE members:', mcb_sims[key])
                print('Control SMYLE members:',intersect_members)
            elif len(ctrl_members)>=len(mcb_members):
                mcb_sims[key] = sorted(mcb_members)
                intersect_members = ctrl_members[:len(mcb_members)]
            print('MCB members:', mcb_sims[key])
            print('Control SMYLE members:',intersect_members)

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
    atm_varnames_monthly_subset = ['SWCF','TS','PRECT','U','V','AWNC','FREQL','PS','QREFHT','CLDLOW','CLDLIQ']
    # Conversion constants
    # PRECT
    m_to_mm = 1e3 #mm/m
    s_to_days = 86400 #s/day

    ## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS
    atm_monthly_drift_clim_xr = fun.reorient_netCDF(xr.open_dataset(glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed/*atm_drift_clim.nc')[0]))

    ## READ IN CONTROL SIMULATION & PRE-PROCESS
    # ATM
    atm_monthly_ctrl[sim]={}
    ts_ctrl_drift[sim]  = {}
    ps_ctrl_drift[sim] = {}
    ts_ctrl_anom[sim]={}
    ps_ctrl_anom[sim]={}
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
        # Calculate BL CDNUMC
        atm_monthly_ctrl[sim][key] = atm_monthly_ctrl[sim][key].assign(CDNUMC=(atm_monthly_ctrl[sim][key]['AWNC']/(1e6)/atm_monthly_ctrl[sim][key]['FREQL']).where(atm_monthly_ctrl[sim][key].lev>850,drop=True).mean(dim='lev'))
        atm_monthly_ctrl[sim][key]['CDNUMC'].attrs['units'] = '#/cm3'
        atm_monthly_ctrl[sim][key]['CDNUMC'].load()
        ##DRIFT CORRECTION
        # Compute drift correction anomaly
        ts_ctrl_drift[sim][key] = atm_monthly_drift_clim_xr['TS'].assign_coords(time=atm_monthly_ctrl[sim][key]['time']).mean(dim='member')
        ps_ctrl_drift[sim][key] = atm_monthly_drift_clim_xr['PS'].assign_coords(time=atm_monthly_ctrl[sim][key]['time']).mean(dim='member')
        # By month climatology
        i_month=np.arange(1,13,1)
        ts_ctrl_anom[sim][key] = atm_monthly_ctrl[sim][key]['TS'] - ts_ctrl_drift[sim][key]
        ps_ctrl_anom[sim][key] = atm_monthly_ctrl[sim][key]['PS'] - ps_ctrl_drift[sim][key]
        ts_ctrl_anom[sim][key].attrs['units']='\N{DEGREE SIGN}C'
        ps_ctrl_anom[sim][key].attrs['units']='Pa'


    ## READ IN MCB SIMULATIONS & PRE-PROCESS
    # ATM
    atm_monthly_mcb[sim]={}
    ts_mcb_anom[sim]={}
    ps_mcb_anom[sim]={}
    for key in mcb_keys:
        atm_monthly_mcb_single_mem = {}
        for m in mcb_sims[key]:
            print(m)
            if (sim=='mcb_100cc') or (sim=='mcb_300cc') or (sim=='mcb_300cc_ST') or (sim=='mcb_300cc_ET') or (sim=='mcb_300cc_ETST') or (sim=='mcb_100cc_AFCDNC'):
                dir_mcb = glob.glob('/_data/MCB/b.e21.BSMYLE.f09_g17.MCB_*'+m+'/atm/proc/tseries/month_1')[0]
            elif (sim=='mcb_500cc'):
                dir_mcb = glob.glob('/_data/MCB/b.e21.BSMYLE.f09_g17.MCB.*'+m+'/atm/proc/tseries/month_1')[0]
            elif sim=='aus':
                dir_mcb = glob.glob('/_data/SMYLE-AUFIRE/b.e21.BSMYLE-AUFIRE.f09_g17.*'+m+'/atm/proc/tseries/month_1')[0]
            file_subset_mcb = []
            for var in atm_varnames_monthly_subset:
                pattern = "."+var+"."
                var_file_mcb = [f for f in os.listdir(dir_mcb) if pattern in f]
                file_subset_mcb.append(dir_mcb+'/'+var_file_mcb[0])
            atm_monthly_mcb_single_mem[m] = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_mfdataset(file_subset_mcb,compat='override')))
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
        # Calculate BL CDNUMC
        atm_monthly_mcb[sim][key] =  atm_monthly_mcb[sim][key].assign(CDNUMC=(atm_monthly_mcb[sim][key]['AWNC']/(1e6)/atm_monthly_mcb[sim][key]['FREQL']).where(atm_monthly_mcb[sim][key].lev>850,drop=True).mean(dim='lev'))
        atm_monthly_mcb[sim][key]['CDNUMC'].attrs['units'] = '#/cm3'
        atm_monthly_mcb[sim][key]['CDNUMC'].load()
        ##DRIFT CORRECTION
        # By month climatology
        i_month=np.arange(1,13,1)
        ts_mcb_anom[sim][key] = atm_monthly_mcb[sim][key]['TS'] - ts_ctrl_drift[sim][key]
        ps_mcb_anom[sim][key] = atm_monthly_mcb[sim][key]['PS'] - ps_ctrl_drift[sim][key]
        ts_mcb_anom[sim][key].attrs['units']='\N{DEGREE SIGN}C'
        ps_mcb_anom[sim][key].attrs['units']='Pa'

    ## Trim length of control to MCB (36 months)
    atm_monthly_ctrl[sim][key] = atm_monthly_ctrl[sim][key].isel(time=slice(None, len(atm_monthly_mcb[sim][key].time)))

    #%% COMPUTE ANOMALIES FOR SELECT VARIABLES
    # Add CDNUMC to varname list
    atm_varnames_monthly_subset = ['CDNUMC','SWCF','TS','PRECT','U','V','QREFHT','CLDLIQ','CLDLOW']
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
    atm_monthly_sig[sim] = {}
    atm_djf_sig[sim] = {}
    atm_mcb_on_sig[sim] = {}
   

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
    
    # Eastern equatorial Pacific SOI
    lat_max = 5
    lat_min = -5
    lon_max = -80
    lon_min = -130
    # Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
    zeros_mask = atm_monthly_ctrl[sim][ctrl_keys[0]].PS.isel(member=0, time=0)*0
    eep_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                    (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                    1,zeros_mask)

    # Western equatorial Pacific SOI
    lat_max = 5
    lat_min = -5
    lon_max = 140
    lon_min = 90
    # Generate box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
    zeros_mask = atm_monthly_ctrl[sim][ctrl_keys[0]].PS.isel(member=0, time=0)*0
    wep_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                    (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                    1,zeros_mask)


    # Identify signficant cells (ensemble mean differences > 2*SE)
    # Calculate standard error of control ensemble
    for key in mcb_keys:
        atm_monthly_sig[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            se =  np.std(atm_monthly_ctrl[sim][ctrl_keys[0]][varname].values,axis=0)/np.sqrt(len(atm_monthly_ctrl[sim][ctrl_keys[0]][varname].member))
            atm_monthly_sig[sim][key][varname] = se # array of 2*SE

    # Calculate standard error of control ensemble for DJF of ENSO event
    atm_varnames_monthly_subset = ['TS']
    for key in mcb_keys:
        atm_djf_sig[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            # Subset first year of simulation
            t1=atm_monthly_ctrl[sim][key][varname].isel(time=slice(7,19))
            # Subset DJF and rename by month
            tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
            tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
            tslice = tslice.rename({'time':'month'})
            tslice = fun.weighted_temporal_mean_clim(tslice)
            # sem = stats.sem(tslice.values,axis=0,nan_policy='omit')
            se =  np.std(tslice,axis=0)/np.sqrt(len(tslice.member))
            # Subset MCB anomaly dataarray for DJF of first year
            t2=atm_monthly_ensemble_anom[sim][key][varname].isel(time=slice(7,19))
            tslice2 =t2.loc[{'time':[t for t in pd.to_datetime(t2.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
            tslice2 =tslice2.assign_coords(time=pd.to_datetime(tslice2.time.values).month)
            tslice2 = tslice2.rename({'time':'month'})
            tslice2 = fun.weighted_temporal_mean_clim(tslice2)
            atm_djf_sig[sim][key][varname] = xr.where(np.abs(tslice2)>2*np.abs(se), 0,1)


    # Calculate standard error of control ensemble for MCB deployment window
    atm_varnames_monthly_subset = ['SWCF']
    for key in mcb_keys:
        atm_mcb_on_sig[sim][key] = {}
        for varname in atm_varnames_monthly_subset:
            print(varname)
            # Subset MCB window
            if (sim=='mcb_100cc') or (sim=='mcb_300cc') or (sim=='mcb_300cc_ST') or (sim=='mcb_300cc_ET') or (sim=='mcb_500cc') or (sim=='mcb_300cc_ETST'):
                mcb_on_start_dict = {'':2,'06-02':1,'06-08':1,'06-11':1,'09-02':4,'09-11':4,'12-02':7}
                mcb_on_end_dict = {'':5,'06-02':10,'06-08':4,'06-11':7,'09-02':10,'09-11':7,'12-02':10}
            elif (same_init=='n') and (sim=='mcb_100cc_AFCDNC'):
                mcb_on_start_dict = {'':1}
                mcb_on_end_dict = {'':4} 
            elif (same_init=='y') and (sim=='mcb_100cc_AFCDNC'):
                mcb_on_start_dict = {'':4}
                mcb_on_end_dict = {'':7} 
            elif sim=='aus':
                mcb_on_start_dict = {'':4} # FOR AFCDNC MASK
                mcb_on_end_dict = {'':7} # FOR AFCDNC MASK  
            tslice=atm_monthly_ctrl[sim][key][varname].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key]))
            tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
            tslice = tslice.rename({'time':'month'})
            tslice = fun.weighted_temporal_mean_clim(tslice)
            # sem = stats.sem(tslice.values,axis=0,nan_policy='omit')
            se =  np.std(tslice,axis=0)/np.sqrt(len(tslice.member))
            # Subset MCB anomaly dataarray for JFM of first year
            tslice2=atm_monthly_ensemble_anom[sim][key][varname].isel(time=slice(mcb_on_start_dict[key],mcb_on_end_dict[key]))
            tslice2 =tslice2.assign_coords(time=pd.to_datetime(tslice2.time.values).month)
            tslice2 = tslice2.rename({'time':'month'})
            tslice2 = fun.weighted_temporal_mean_clim(tslice2)
            atm_mcb_on_sig[sim][key][varname] = xr.where(np.abs(tslice2)>2*np.abs(se), 0,1)

    ## Calculate Niño3.4 index
    # Control
    nino34_ctrl[sim]={}
    nino34_ctrl_sem[sim] = {}
    for key in ctrl_keys:
        nino34_ctrl[sim][key] = fun.calc_weighted_mean_tseries(ts_ctrl_anom[sim][key].mean(dim='member').where(nino34_mask>0,drop=True))
        nino34_ctrl_sem[sim][key] = fun.calc_weighted_mean_tseries(2*(ts_ctrl_anom[sim][key]).std(dim='member').where(nino34_mask>0,drop=True)/np.sqrt(len(ts_ctrl_anom[sim][key].member)))
    # MCB
    nino34_mcb[sim]={}
    nino34_mcb_sem[sim] = {}
    for key in mcb_keys:
        nino34_mcb[sim][key] = fun.calc_weighted_mean_tseries(ts_mcb_anom[sim][key].mean(dim='member').where(nino34_mask>0,drop=True))
        nino34_mcb_sem[sim][key] = fun.calc_weighted_mean_tseries(2*(ts_mcb_anom[sim][key]).std(dim='member').where(nino34_mask>0,drop=True)/np.sqrt(len(ts_mcb_anom[sim][key].member)))

    ## Calculate Niño4 index
    # Control
    nino4_ctrl[sim]={}
    nino4_ctrl_sem[sim] = {}    
    for key in ctrl_keys:
        nino4_ctrl[sim][key] = fun.calc_weighted_mean_tseries(ts_ctrl_anom[sim][key].mean(dim='member').where(nino4_mask>0,drop=True))
        nino4_ctrl_sem[sim][key] = fun.calc_weighted_mean_tseries(2*(ts_ctrl_anom[sim][key]).std(dim='member').where(nino4_mask>0,drop=True)/np.sqrt(len(ts_ctrl_anom[sim][key].member)))
    # MCB
    nino4_mcb[sim]={}
    nino4_mcb_sem[sim] = {}
    for key in mcb_keys:
        nino4_mcb[sim][key] = fun.calc_weighted_mean_tseries(ts_mcb_anom[sim][key].mean(dim='member').where(nino4_mask>0,drop=True))
        nino4_mcb_sem[sim][key] = fun.calc_weighted_mean_tseries(2*(ts_mcb_anom[sim][key]).std(dim='member').where(nino4_mask>0,drop=True)/np.sqrt(len(ts_mcb_anom[sim][key].member)))

    ## Calculate SOI
    # Control
    soi_ctrl[sim]={}
    soi_ctrl_sem[sim] = {}
    for key in ctrl_keys:
        soi_ctrl[sim][key],soi_ctrl_sem[sim][key] = calc_soi(ps_ctrl_anom[sim][key])
    # MCB
    soi_mcb[sim]={}
    soi_mcb_sem[sim] = {}
    for key in mcb_keys:
        soi_mcb[sim][key],soi_mcb_sem[sim][key] = calc_soi(ps_mcb_anom[sim][key])


### PLOT NIÑO3.4 ABS
## AUFIRE
sim='aus'
plt.figure(figsize=(7,4));
# CONTROL
plot_xr = nino34_ctrl[sim][key]-nino34_ctrl[sim][key].isel(time=0)
plot_sem_xr = nino34_ctrl_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr, plot_xr + plot_sem_xr,color='k', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='k',linewidth=3,label='Control');
# AUS
plot_xr = nino34_mcb[sim][key]-nino34_mcb[sim][key].isel(time=0)
plot_sem_xr = nino34_mcb_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr,plot_xr + plot_sem_xr,color='#b2182b', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='#b2182b',linewidth=3,label='AUFIRE');
## MCB
sim='mcb_100cc_AFCDNC'
# MCB
plot_xr = nino34_mcb[sim][key]-nino34_mcb[sim][key].isel(time=0)
plot_sem_xr = nino34_mcb_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr,plot_xr + plot_sem_xr,color='#2166ac', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='#2166ac',linewidth=3,label='MCB');
## FIGURE AESTHETICS
plt.legend();
# Format dates
ax=plt.gca();
xbounds=ax.get_xlim();
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
# Add axes labels
plt.xlabel('Time',fontsize=12); plt.ylabel('Niño 3.4 SST anomaly (\N{DEGREE SIGN}C)', fontsize=12);
# Add y=0 line
plt.axhline(y=0,color='grey',linestyle='dotted');
plt.tight_layout();


### PLOT NIÑO4 ABS
## AUFIRE
sim='aus'
plt.figure(figsize=(7,4));
# CONTROL
plot_xr = nino4_ctrl[sim][key]-nino4_ctrl[sim][key].isel(time=0)
plot_sem_xr = nino4_ctrl_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr, plot_xr + plot_sem_xr,color='k', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='k',linewidth=3,label='Control');
# AUS
plot_xr = nino4_mcb[sim][key]-nino4_mcb[sim][key].isel(time=0)
plot_sem_xr = nino4_mcb_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr,plot_xr + plot_sem_xr,color='#b2182b', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='#b2182b',linewidth=3,label='AUFIRE');
## MCB
sim='mcb_100cc_AFCDNC'
# MCB
plot_xr = nino4_mcb[sim][key]-nino4_mcb[sim][key].isel(time=0)
plot_sem_xr = nino4_mcb_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr,plot_xr + plot_sem_xr,color='#2166ac', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='#2166ac',linewidth=3,label='MCB');
## FIGURE AESTHETICS
plt.legend();
# Format dates
ax=plt.gca();
xbounds=ax.get_xlim();
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
# Add axes labels
plt.xlabel('Time',fontsize=12); plt.ylabel('Niño 4 SST anomaly (\N{DEGREE SIGN}C)', fontsize=12);
# Add y=0 line
plt.axhline(y=0,color='grey',linestyle='dotted');
plt.tight_layout();


### PLOT SOI ABS
sim='aus'
plt.figure(figsize=(7,4));
# CONTROL
plot_xr = soi_ctrl[sim][key]-soi_ctrl[sim][key].isel(time=0)
plot_sem_xr = soi_ctrl_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr, plot_xr + plot_sem_xr,color='k', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='k',linewidth=3,label='Control');
# AUS
plot_xr = soi_mcb[sim][key]-soi_mcb[sim][key].isel(time=0)
plot_sem_xr = soi_mcb_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr,plot_xr + plot_sem_xr,color='#b2182b', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='#b2182b',linewidth=3,label='AUFIRE');
## MCB
sim='mcb_100cc_AFCDNC'
# MCB
plot_xr = soi_mcb[sim][key]-soi_mcb[sim][key].isel(time=0)
plot_sem_xr = soi_mcb_sem[sim][key]
# PLOT 2 STANDARD ERRORS
plt.fill_between(plot_xr.time, plot_xr - plot_sem_xr,plot_xr + plot_sem_xr,color='#2166ac', alpha=0.2)
# PLOT ENSEMBLE MEAN
plt.plot(plot_xr.time,plot_xr,color='#2166ac',linewidth=3,label='MCB');
## FIGURE AESTHETICS
plt.legend();
# Format dates
ax=plt.gca();
xbounds=ax.get_xlim();
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# Rotates and right-aligns the x labels so they don't crowd each other.
for label in ax.get_xticklabels(which='major'):
    label.set(rotation=30, horizontalalignment='right')
# Add axes labels
plt.xlabel('Time',fontsize=12); plt.ylabel('SOI', fontsize=12);
# Add y=0 line
plt.axhline(y=0,color='grey',linestyle='dotted');
plt.tight_layout();



## SWCF, CLDLIQ, 2m Q, TS seeding region time series
varlist = ['SWCF','CLDLIQ','QREFHT','TS']
sesp_region = fun.reorient_netCDF(xr.open_dataset('/_data/sesp_mask_CESM2_0.9x1.25_v3.nc')).mask.isel(time=7)
tseries = {}
tseries_se = {}
for sim in sim_keys:
    tseries[sim]={}
    tseries_se[sim]={}
    for key in mcb_keys:
        tseries[sim][key]={}
        tseries_se[sim][key]={}
        for var in varlist:
            print(var)
            if var=='CLDLIQ':
                # Sum from surface to ~850 hPa
                tseries[sim][key][var] = atm_monthly_anom[sim][key][var].where((sesp_region>0)&(atm_monthly_anom[sim][key][var].lev>850),drop=True).sum(dim='lev')
            else:
                tseries[sim][key][var] = atm_monthly_anom[sim][key][var].where(sesp_region>0,drop=True)
            # Calculate area-weighted mean over the SESP region
            tseries[sim][key][var] = fun.calc_weighted_mean_tseries(tseries[sim][key][var])
            tseries[sim][key][var].attrs['units'] = atm_monthly_anom[sim][key][var].units
            tseries_se[sim][key][var] =  2*np.std(tseries[sim][key][var],axis=0)/np.sqrt(len(tseries[sim][key][var].member))
            tseries[sim][key][var].load()
            tseries_se[sim][key][var].load()


## Plot time series
fig,axs=plt.subplots(2,2,figsize=(12,6),sharex=True);
subplot_label = list(string.ascii_uppercase)
subplot_num=1
colormap = {'aus':'#b2182b', 'mcb_100cc_AFCDNC':'#2166ac'}
labelmap = {'aus':'AUFIRE', 'mcb_100cc_AFCDNC':'MCB'}
long_name = {'SWCF':'Shortwave cloud forcing', 'CLDLIQ':'Cloud liquid water path','QREFHT':'Specific humidity','TS':'Surface temperature'}
for var in varlist:
    plt.subplot(2,2,subplot_num);
    for sim in sim_keys:
        plot_xr = tseries[sim][key][var] - tseries[sim][key][var].isel(time=0)
        plot_sem_xr = tseries_se[sim][key][var]
        # PLOT 2 STANDARD ERRORS
        plt.fill_between(plot_xr.time, plot_xr.mean(dim='member') - plot_sem_xr,plot_xr.mean(dim='member') + plot_sem_xr,color=colormap[sim], alpha=0.2)
        # PLOT ENSEMBLE MEAN
        plt.plot(plot_xr.time,plot_xr.mean(dim='member'),color=colormap[sim],linewidth=3,label=labelmap[sim]);
    ## SUBPLOT AESTHETICS
    # Add MCB window
    plt.axvline(plot_xr.time[mcb_on_start_dict['']].values,c='k',linestyle='--',linewidth=2);
    plt.axvline(plot_xr.time[mcb_on_end_dict['']].values,c='k',linestyle='--',linewidth=2);
    # Format dates
    ax=plt.gca();
    xbounds=ax.get_xlim();
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    # Rotates and right-aligns the x labels so they don't crowd each other.
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    # Add axes labels
    if subplot_num>2:
        plt.xlabel('Time',fontsize=12); 
    elif subplot_num==2:
        plt.legend();
    if var=='SWCF':
         plt.ylabel(long_name[var]+' (W/m$^{2}$)', fontsize=12);
    else:
        plt.ylabel(long_name[var]+' ('+tseries[sim][key][var].units+')', fontsize=12);
    # Add y=0 line
    plt.axhline(y=0,color='grey',linestyle='dotted');
    # Add subplot label
    plt.title(subplot_label[subplot_num-1],fontweight='bold',fontsize=14,loc='left',pad=15)
    subplot_num+=1
# Align y-labels
fig.align_ylabels();
plt.tight_layout();



## DJF Zonal mean precipitation over Pacific and Indian ocean
## Read in SMYLE ocean output file to get region mask
ocn_file = fun.reorient_netCDF(xr.open_dataset('/_data/MCB/ocn_processed/2015-05_06-02.2015-05/b.e21.BSMYLE.f09_g17.2015-05.TEMP.101-110.nc'))['REGION_MASK'].isel(member=0)
ocn_file = ocn_file.assign_coords(lat= atm_monthly_ctrl[sim][ctrl_keys[0]].lat, lon= atm_monthly_ctrl[sim][ctrl_keys[0]].lon)
## Define Pacific region (REGION_MASK=2)
pacific_mask = xr.where(ocn_file==2, 1,np.nan)
## Define Pacific region (REGION_MASK=3)
io_mask = xr.where(ocn_file==3, 1,np.nan)

pacific_zonal_mean_ctrl = {}
io_zonal_mean_ctrl = {}
pacific_zonal_mean_mcb = {}
io_zonal_mean_mcb = {}

for sim in sim_keys:
    ## MCB
    in_xr = atm_monthly_mcb[sim]['']['PRECT']
    # Subset first year of simulation
    t1=in_xr.isel(time=slice(7,19))
    # Subset DJF
    tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    # Create Pacific Ocean zonal mean
    pacific_zonal_mean_mcb[sim] = tslice.where(pacific_mask==1, drop=True).mean(dim=('member','lon','time'))
    # Create Indian Ocean zonal mean
    io_zonal_mean_mcb[sim] = tslice.where(io_mask==1, drop=True).mean(dim=('member','lon','time'))
    print('Pacific (',sim,'):',float(pacific_zonal_mean_mcb[sim].where(pacific_zonal_mean_mcb[sim]==pacific_zonal_mean_mcb[sim].max(),drop=True).lat.values),' degrees')
    print('Indian (',sim,'):',float(io_zonal_mean_mcb[sim].where(io_zonal_mean_mcb[sim]==io_zonal_mean_mcb[sim].max(),drop=True).lat.values),' degrees')

    ## Control
    in_xr = atm_monthly_ctrl[sim]['']['PRECT']
    # Subset first year of simulation
    t1=in_xr.isel(time=slice(7,19))
    # Subset DJF
    tslice=t1.loc[{'time':[t for t in pd.to_datetime(t1.time.values) if (t.month==12)|(t.month==1)|(t.month==2)]}]
    # Create Pacific Ocean zonal mean
    pacific_zonal_mean_ctrl[sim] = tslice.where(pacific_mask==1, drop=True).mean(dim=('member','lon','time'))
    # Create Indian Ocean zonal mean
    io_zonal_mean_ctrl[sim] = tslice.where(io_mask==1, drop=True).mean(dim=('member','lon','time'))
    # Print latitude of maximum precipitation
    print('Pacific (control):',float(pacific_zonal_mean_ctrl[sim].where(pacific_zonal_mean_ctrl[sim]==pacific_zonal_mean_ctrl[sim].max(),drop=True).lat.values),' degrees')
    print('Indian (control):',float(io_zonal_mean_ctrl[sim].where(io_zonal_mean_ctrl[sim]==io_zonal_mean_ctrl[sim].max(),drop=True).lat.values),' degrees')


## Plot zonal means for Pacific and Indian Ocean precipitation
# Create figure
fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True);
subplot_label = list(string.ascii_uppercase)
subplot_num=1
colormap = {'aus':'#b2182b', 'mcb_100cc_AFCDNC':'#2166ac'}
labelmap = {'aus':'AUFIRE', 'mcb_100cc_AFCDNC':'MCB'}
long_name = {'SWCF':'Shortwave cloud forcing', 'CLDLIQ':'Cloud liquid water path','QREFHT':'Specific humidity','TS':'Surface temperature'}
# Pacific Ocean
plt.subplot(2,1,subplot_num);
# Add control
plt.plot(pacific_zonal_mean_ctrl['aus'].lat,pacific_zonal_mean_ctrl['aus'].values,color='k',linestyle='--',linewidth=3,label='Control')
# Add MCB
for sim in sim_keys:    
    plt.plot(pacific_zonal_mean_mcb[sim].lat,pacific_zonal_mean_mcb[sim].values,color=colormap[sim],linewidth=3,label=labelmap[sim])
# Set xlim to tropics
# plt.xlim(-15,15);
# Add legend
plt.legend();
# Add axis labels
plt.ylabel('Precipitation (mm/day)',fontsize=12);
# Add annotation for Pacific Ocean
plt.annotate('Pacific', xy=(0.01,0.05), ha='left', xycoords='axes fraction',color='k',fontsize=12);
# Add subplot label
plt.title(subplot_label[subplot_num-1],fontweight='bold',fontsize=14,loc='left')
subplot_num+=1
# Indian Ocean
plt.subplot(2,1,subplot_num);
# Add control
plt.plot(io_zonal_mean_ctrl['aus'].lat,io_zonal_mean_ctrl['aus'].values,color='k',linestyle='--',linewidth=3,label='Control')
# Add MCB
for sim in sim_keys:    
    plt.plot(io_zonal_mean_mcb[sim].lat,io_zonal_mean_mcb[sim].values,color=colormap[sim],linewidth=3,label=labelmap[sim])
# Set xlim to tropics
plt.xlim(-10,10);
# Add legend
# plt.legend();
# Add axis labels
plt.ylabel('Precipitation (mm/day)',fontsize=12);
plt.xlabel('Latitude (degrees)',fontsize=12)
# Add annotation for Indian Ocean
plt.annotate('Indian', xy=(0.01,0.05), ha='left', xycoords='axes fraction',color='k',fontsize=12);
# Add subplot label
plt.title(subplot_label[subplot_num-1],fontweight='bold',fontsize=14,loc='left')
plt.tight_layout();





#%% PLOT MCB WINDOW SWCF, DJF PRECT, AND DJF TS MAPS (3X3)
plot_labels = ['A','B','C','D','E','F']
sim_pairs = [['aus','mcb_100cc_AFCDNC']]
cldrf_opt = 'SWCF'

for pair in sim_pairs:
    fig = plt.figure(figsize=(12,4));
    subplot_num = 0
    # Get overlay mask files
    seeding_mask = fun.reorient_netCDF(xr.open_dataset('/_data/sesp_mask_CESM2_0.9x1.25_v19.nc')).mask.isel(time=[-1,0,1]).mean(dim='time')
    # Force seeding mask lat, lon to equal the output CESM2 data (rounding errors)
    seeding_mask_seed = seeding_mask.assign_coords({'lat':atm_monthly_ctrl[pair[-1]][ctrl_keys[0]]['lat'], 'lon':atm_monthly_ctrl[pair[-1]][ctrl_keys[0]]['lon']})

    # Subset 1 month of seeded grid cells 
    # Add cyclical point for ML 
    seeding_mask_seed_wrap, lon_wrap = add_cyclic_point(seeding_mask_seed,coord=seeding_mask_seed.lon)
    for sim in pair:
        cmin=-40
        cmax=40
        ## Calculate the MCB mean for the first simulated year of the simulation
        # Subset MCB period
        if (sim=='mcb_100cc') or (sim=='mcb_300cc') or (sim=='mcb_300cc_ST') or (sim=='mcb_300cc_ET') or (sim=='mcb_500cc')or (sim=='mcb_300cc_ETST'):
            label = {'mcb_100cc': '100 #/cm$^{3}$','mcb_300cc': '300 #/cm$^{3}$','mcb_500cc': '500 #/cm$^{3}$','mcb_300cc_ST':'300 #/cm$^{3}$','mcb_300cc_ET':'300 #/cm$^{3}$','mcb_300cc_ETST':'300 #/cm$^{3}$'}
            tslice=atm_monthly_ensemble_anom[sim][''][cldrf_opt].isel(time=slice(2,5))
        elif (sim=='mcb_100cc_AFCDNC') and (same_init=='n'):
            tslice=atm_monthly_ensemble_anom[sim][''][cldrf_opt].isel(time=slice(1,4))
            label = {'mcb_100cc_AFCDNC': '100 #/cm$^{3}$'}
        elif (sim=='mcb_100cc_AFCDNC') and (same_init=='y'):
            tslice=atm_monthly_ensemble_anom[sim][''][cldrf_opt].isel(time=slice(4,7))
            label = {'mcb_100cc_AFCDNC': '100 #/cm$^{3}$'}
        elif sim=='aus':
            tslice=atm_monthly_ensemble_anom[sim][''][cldrf_opt].isel(time=slice(4,7))
            label={'aus':'AUFIRE'}
        tlabel='MCB window'
        tslice=tslice.assign_coords(time=pd.to_datetime(tslice.time.values).month)
        tslice = tslice.rename({'time':'month'})
        # Calculate weighted temporal mean and assign units
        in_xr = fun.weighted_temporal_mean_clim(tslice)
        in_xr.attrs['units'] = 'W/m2'

        # Get mean value in seeding region for plot
        regional_val = float(fun.calc_weighted_mean_tseries(in_xr.where(seeding_mask_seed>0,drop=True)).values)
        global_val = float(fun.calc_weighted_mean_tseries(in_xr).values)
        summary_stat = [regional_val, global_val]
        print(sim, 'MCB region:',summary_stat[0],'W/m2') #print values for main text
        print(sim, 'Global:',summary_stat[1],'W/m2') #print values for main text
        swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='bwr', plot_zoom='global', central_lon=180,\
                                CI_in=atm_mcb_on_sig[sim][''][cldrf_opt],CI_level=0.05,CI_display='inv_stipple',\
                                projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=False)
        plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='grey', linewidths=.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
        ## Add experiment labels to first column
        if sim=='aus':
            plt.annotate('2020-21 La Niña + wildfires', xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
        else:
            plt.annotate('2020-21 La Niña + MCB ('+label[sim]+')', xy=(0.5,1.02), ha='center', xycoords='axes fraction',color='k');
        subplot_num+=1
        
        # DJF PRECT
        lev_sfc = float(atm_monthly_mcb[sim][key].lev[-1].values)
        cmin=-2
        cmax=2
        ## Calculate the DJF mean for the first simulated year of the simulation
        # Subset first year of simulation
        t1=atm_monthly_ensemble_anom[sim]['']['PRECT'].isel(time=slice(7,19))
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
                                projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=False)
        m1 = plt.quiver(atm_monthly_mcb[sim][key]['U'].lon.values[::10], atm_monthly_mcb[sim][key]['U'].lat.values[::10],\
                    atm_monthly_mcb[sim][key].mean(dim='member').isel(time=8).sel(lev=lev_sfc)['U'].values[::10,::10],atm_monthly_mcb[sim][key].mean(dim='member').isel(time=8).sel(lev=lev_sfc)['V'].values[::10,::10],\
                    transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.0025);
        plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='grey', linewidths=.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
        # Add Nino3.4 mean value as annotation
        subplot_num+=1

        # DJF TS
        t1=atm_monthly_ensemble_anom[sim]['']['TS'].isel(time=slice(7,19))
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
        print(sim, 'nino3.4:',nino34_mean_val,'deg C') # print values for main text
        print(sim, 'global:',global_mean_val,'deg C') # print values for main text
        summary_stat = [mcb_mean_val, np.nan]
        ts, p3 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap='RdBu_r', plot_zoom='global', central_lon=180,\
                                CI_in=ci_in,CI_level=ci_level,CI_display=ci_display,\
                                projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=False)
        plt.contour(lon_wrap,seeding_mask_seed.lat,seeding_mask_seed_wrap,\
                transform= ccrs.PlateCarree(),levels=np.linspace(0,1,2), colors='grey', linewidths=.5,add_colorbar=False,\
                subplot_kws={'projection':ccrs.Robinson(central_longitude=180)});
        plt.title(plot_labels[subplot_num],fontsize=14, fontweight='bold',loc='left');
        subplot_num+=1
    fig.subplots_adjust(bottom=0.1, top=0.95, wspace=0.1,hspace=0.1);
    ## Add PRECT quiver key
    plt.quiverkey(m1, X=6.2, Y=0.82, U= 10, label='10 ms$^{-1}$', labelpos='E', coordinates = 'inches');
    ## Add colorbars to bottom of figure
    cbar_ax = fig.add_axes([0.12, 0.07, 0.25, 0.025]) #rect kwargs [left, bottom, width, height];
    if cldrf_opt=='SWCF':
        plt.colorbar(p1, cax = cbar_ax, orientation='horizontal', label='SW radiative forcing (W/m$^{2}$)', extend='both',pad=0.1);
    elif cldrf_opt=='CLALB':
        plt.colorbar(p1, cax = cbar_ax, orientation='horizontal', label='Cloud albedo', extend='both',pad=0.1);
    cbar_ax = fig.add_axes([0.385, 0.07, 0.25, 0.025]) #rect kwargs [left, bottom, width, height];
    plt.colorbar(p2, cax = cbar_ax, orientation='horizontal', label='Precipitation (mm/day)', extend='both',pad=0.1);
    cbar_ax = fig.add_axes([0.655, 0.07, 0.25, 0.025]) #rect kwargs [left, bottom, width, height];
    plt.colorbar(p3, cax = cbar_ax, orientation='horizontal', label='Temperature (\N{DEGREE SIGN}C)', extend='both',pad=0.1);


#%% PLOT CDNUMC FOR AUFIRE
# Mask out land
# Specify ocean/land fraction file directory and read in mask
landfrac_wd = '/_data/'
landfrac = fun.reorient_netCDF(xr.open_dataset(landfrac_wd+'cesm2_landfrac_0.9x1.25.nc')).LANDFRAC #1 for land, 0 for ocean grid boxes

atm_monthly_mcb['aus']['']['CDNUMC_ocn'] = xr.where(landfrac==0,atm_monthly_mcb['aus']['']['CDNUMC'],np.nan)
atm_monthly_mcb['aus']['']['CDNUMC_ocn'].attrs['units'] = '#/cm$^{3}$'
atm_monthly_ensemble_anom['aus']['']['CDNUMC_ocn'] = xr.where(landfrac==0,atm_monthly_ensemble_anom['aus']['']['CDNUMC'],np.nan)
atm_monthly_ensemble_anom['aus']['']['CDNUMC_ocn'].attrs['units'] = '#/cm$^{3}$'
atm_monthly_sig['aus']['']['CDNUMC_ocn'] = xr.where(landfrac==0,atm_monthly_sig['aus']['']['CDNUMC'],np.nan)
atm_monthly_sig['aus']['']['CDNUMC_ocn'].attrs['units'] = '#/cm$^{3}$'


## Create figure
fig = plt.figure(figsize=(12,6));
subplot_num = 0
plot_labels = ['A','B','C','D','E','F']

## AUFIRE (raw values)
cmin=0
cmax=100
cmap='inferno'

# Dec 2019
itime=4
in_xr = atm_monthly_mcb['aus']['']['CDNUMC_ocn'].isel(time=itime).mean(dim='member')
in_xr.attrs['units'] = '#/cm$^{3}$'
swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap=cmap, plot_zoom='global', central_lon=180,\
                        #CI_in=sig_xr,CI_level=0.05,CI_display='inv_stipple',\
                        projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=True)
plt.annotate(str(in_xr.time.dt.strftime('%Y-%m').values), xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
plt.title(plot_labels[subplot_num],fontsize=12, fontweight='bold',loc='left');
subplot_num+=1
# Jan 2020
itime+=1
in_xr = atm_monthly_mcb['aus']['']['CDNUMC_ocn'].isel(time=itime).mean(dim='member')
in_xr.attrs['units'] = '#/cm$^{3}$'
swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap=cmap, plot_zoom='global', central_lon=180,\
                        #CI_in=sig_xr,CI_level=0.05,CI_display='inv_stipple',\
                        projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=True)
plt.annotate(str(in_xr.time.dt.strftime('%Y-%m').values), xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
plt.title(plot_labels[subplot_num],fontsize=12, fontweight='bold',loc='left');
subplot_num+=1
# Feb 2020
itime+=1
in_xr = atm_monthly_mcb['aus']['']['CDNUMC_ocn'].isel(time=itime).mean(dim='member')
in_xr.attrs['units'] = '#/cm$^{3}$'
swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap=cmap, plot_zoom='global', central_lon=180,\
                        #CI_in=sig_xr,CI_level=0.05,CI_display='inv_stipple',\
                        projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=True)
plt.annotate(str(in_xr.time.dt.strftime('%Y-%m').values), xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
plt.title(plot_labels[subplot_num],fontsize=12, fontweight='bold',loc='left');
subplot_num+=1


## AUFIRE-CTRL
cmin=-100
cmax=100
cmap='RdYlBu_r'

# Dec 2019
itime=4
in_xr = atm_monthly_ensemble_anom['aus']['']['CDNUMC_ocn'].isel(time=itime)
sig_xr = xr.where(in_xr>in_xr.quantile(0.85), 0,1)
swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap=cmap, plot_zoom='global', central_lon=180,\
                        CI_in=sig_xr,CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=True)
plt.annotate(str(in_xr.time.dt.strftime('%Y-%m').values), xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
plt.title(plot_labels[subplot_num],fontsize=12, fontweight='bold',loc='left');
subplot_num+=1
# Jan 2020
itime+=1
in_xr = atm_monthly_ensemble_anom['aus']['']['CDNUMC_ocn'].isel(time=itime)
sig_xr = xr.where(in_xr>in_xr.quantile(0.85), 0,1)
swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap=cmap, plot_zoom='global', central_lon=180,\
                        CI_in=sig_xr,CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=True)
plt.annotate(str(in_xr.time.dt.strftime('%Y-%m').values), xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
plt.title(plot_labels[subplot_num],fontsize=12, fontweight='bold',loc='left');
subplot_num+=1
# Feb 2020
itime+=1
in_xr = atm_monthly_ensemble_anom['aus']['']['CDNUMC_ocn'].isel(time=itime)
sig_xr = xr.where(in_xr>in_xr.quantile(0.85), 0,1)
swcf, p1 = fun.plot_panel_maps(in_xr=in_xr, cmin=cmin, cmax=cmax, ccmap=cmap, plot_zoom='global', central_lon=180,\
                        CI_in=sig_xr,CI_level=0.05,CI_display='mask',\
                        projection='Robinson',nrow=2,ncol=3,subplot_num=subplot_num,mean_val='none',cbar=True)
plt.annotate(str(in_xr.time.dt.strftime('%Y-%m').values), xy=(.5,1.02), ha='center', xycoords='axes fraction',color='k');
plt.title(plot_labels[subplot_num],fontsize=12, fontweight='bold',loc='left');
subplot_num+=1