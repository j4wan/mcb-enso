### PURPOSE: Script to compare net surface heat flux anomaly and upper ocean heat content for SMYLE/MCB simulations
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

plt.ion();

dask.config.set({"array.slicing.split_large_chunks": False})

##################################################################################################################
## WHICH EXPERIMENT ARE YOU READING IN? ##
month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
year_init = input('Which initialization year are you reading in (1997, 2015, 2019?): ')
enso_phase = input('Which ENSO event are you reading in (nino or nina)?: ')
sensitivity_opt = input('Sensitivity run (y or n)?: ')
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
ocn_varnames_monthly_subset = ['TAUX','TEMP','SHF']


## READ IN CONTROL SIMULATION & PRE-PROCESS
# OCN
ocn_monthly_ctrl={}
temp_ctrl_anom={}
temp_ctrl_anom_std={}
temp_ctrl_anom_sem={}
ctrl_keys=['']

for key in ctrl_keys:
    path = os.path.join('/_data/realtime/ocn_processed')
    # Read in ensemble and time averaged data
    ocn_monthly_ctrl[key] = fun.reorient_netCDF(xr.open_dataset(path+'/b.e21.BSMYLE.f09_g17.2015-05.pop.combined.'+intersect_members[0][-3:]+'-'+intersect_members[-1][-3:]+'.nc'))
    # Unit correction
    # Convert depth from cm to m
    ocn_monthly_ctrl[key]['z_t'] = ocn_monthly_ctrl[key]['z_t']/100
    ocn_monthly_ctrl[key]['z_t'].attrs['units'] = 'm'
    # Convert TAUX dyne/cm2 to Pa
    ocn_monthly_ctrl[key]['TAUX'] = ocn_monthly_ctrl[key]['TAUX']*(100**2)
    ocn_monthly_ctrl[key]['TAUX'].attrs['units'] = 'Pa'


## READ IN MCB SIMULATIONS & PRE-PROCESS
# OCN
ocn_monthly_mcb={}
temp_mcb_anom={}
temp_mcb_anom_std={}
temp_mcb_anom_sem={}
for key in mcb_keys:
    path = os.path.join('/_data/MCB/ocn_processed')
    # Read in ensemble and time averaged data
    ocn_monthly_mcb[key] = fun.reorient_netCDF(xr.open_dataset(path+'/b.e21.BSMYLE.f09_g17.2015-05.pop.combined.'+mcb_sims[key][0][-3:]+'-'+mcb_sims[key][-1][-3:]+'.nc'))
    # Unit correction
    # Convert depth from cm to m
    ocn_monthly_mcb[key]['z_t'] = ocn_monthly_mcb[key]['z_t']/100
    ocn_monthly_mcb[key]['z_t'].attrs['units'] = 'm'
    # Convert TAUX dyne/cm2 to Pa
    ocn_monthly_mcb[key]['TAUX'] = ocn_monthly_mcb[key]['TAUX']*(100**2)
    ocn_monthly_mcb[key]['TAUX'].attrs['units'] = 'Pa'


#%% COMPUTE ANOMALIES FOR SELECT VARIABLES
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


#%% DEFINE MASKS
## Define Niño 3.4 region
lat_max = 5
lat_min = -5
lon_max = -120
lon_min = -170
# Generate Niño 3.4 box with lat/lon bounds and ocean grid cells only consisting of 1s and 0s
zeros_mask = ocn_monthly_ctrl[ctrl_keys[0]].SHF.isel(member=0, time=0)*0
nino34_mask = xr.where((zeros_mask.lat>=lat_min) & (zeros_mask.lat<=lat_max) &\
                                (zeros_mask.lon>=lon_min) & (zeros_mask.lon<=lon_max),\
                                1,zeros_mask)
# Add cyclical point for ML 
nino34_mask_wrap, lon_wrap = add_cyclic_point(nino34_mask,coord=nino34_mask.lon)


#%% CALCULATE TIME AND AREA INTEGRATED SURFACE HEAT FLUX AND CHANGE IN UPPER OCEAN HEAT CONTENT (<100 m)
# Calculate grid cell area
globalarea_xr = fun.globalarea(zeros_mask)*1e3 #convert to m
# Create empty arrays to populate
shf_anom_sum = {}
ohc_anom_sum = {}
heat_residual = {}
shf_anom_sum_std = {}
ohc_anom_sum_std = {}
heat_residual_std = {}
# Define constants
cp_seawater = 3990 #J/kg/C
rho_seawater = 1026 #kg/m3
for key in mcb_keys:
    print(key)
    ## 1) SURFACE HEAT FLUX
    # Multiply SHF by grid cell area
    sh_tmp = ocn_monthly_anom[key]['SHF']*globalarea_xr
    # Mask Niño3.4 region
    sh_nino34_tmp = sh_tmp.where((nino34_mask==1),drop=True)
    # Take the area-weighted sum
    sh_nino34_area_sum_tmp = fun.calc_area_weighted_sum(sh_nino34_tmp)
    # Convert s to days and multiply by days per month
    s_to_days = 86400 #s/day
    month_day_dict = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    for mon in list(month_day_dict.keys()):
        tmp_month_day = month_day_dict[mon]
        sh_nino34_area_sum_tmp.loc[{'time':[t for t in pd.to_datetime(sh_nino34_area_sum_tmp.time.values) if t.month==mon]}]*=s_to_days*tmp_month_day
    # Subset DJF add ENSO peak
    if year_init=='1997':
        peak_yrs = [1997,1998]
    elif year_init=='2015':
        peak_yrs = [2015,2016]
    # Subset June t to Feb t+1 
    shf_nino34_area_sum_tmp_slice = sh_nino34_area_sum_tmp.isel(time=slice(1,10))
    # Integrate over time
    shf_anom_sum[key] = float(shf_nino34_area_sum_tmp_slice.sum(dim='time').mean(dim='member'))
    # Calculate standard deviation
    shf_anom_sum_std[key] = float(shf_nino34_area_sum_tmp_slice.sum(dim='time').std(dim='member'))

    ## 2) OCEAN HEAT CONTENT
    # Multiply potential temperature by specific heat capacity, density and area to obtain total heat over the vertical column
    ohc_tmp = ocn_monthly_anom[key]['TEMP']*cp_seawater*rho_seawater*globalarea_xr
    # Mask Niño3.4 region
    ohc_nino34_tmp = ohc_tmp.where((nino34_mask==1),drop=True)
    # Take the area-weighted sum
    ohc_nino34_area_sum_tmp = fun.calc_area_weighted_sum(ohc_nino34_tmp)
    # Take the sum over the upper 100 m
    ohc_nino34_area_depth_sum_tmp = (ohc_nino34_area_sum_tmp.where(ohc_nino34_area_sum_tmp.z_t<100,drop=True)*10).sum(dim='z_t')
    # Subset June t to Feb t+1 
    ohc_nino34_area_sum_tmp_slice = ohc_nino34_area_depth_sum_tmp.isel(time=slice(1,10))
    # Average over time
    ohc_anom_sum[key] = float(fun.weighted_temporal_mean(ohc_nino34_area_sum_tmp_slice, by_year=False).mean(dim='member'))
    # Calculate standard deviation
    ohc_anom_sum_std[key] = float(fun.weighted_temporal_mean(ohc_nino34_area_sum_tmp_slice, by_year=False).std(dim='member'))

    ## 3) HEAT RESIDUAL
    heat_residual[key] = float(ohc_anom_sum[key]- shf_anom_sum[key])
    heat_residual_std[key] = float((shf_nino34_area_sum_tmp_slice.sum(dim='time')-fun.weighted_temporal_mean(ohc_nino34_area_sum_tmp_slice, by_year=False)).std(dim='member'))

# Reformat output into dataframe for plotting
heat_budget_df = pd.DataFrame({'Experiment':shf_anom_sum.keys(),'SHF':list(shf_anom_sum.values()),'SHF_std':list(shf_anom_sum_std.values()),\
                               'OHC':list(ohc_anom_sum.values()),'OHC_std':list(ohc_anom_sum_std.values()),\
                                'Residual':list(heat_residual.values()),'Residual_std':list(heat_residual_std.values())})
# Add year column
heat_budget_df['Year'] = year_init
# Print data frame
print(heat_budget_df)
# Export as .csv to plot both 1997/98 and 2015/16 events on same axes
heat_budget_df.to_csv('/_data/heat_budget/heat_budget_enso_'+year_init+'_v2.csv')

# Once the above block is run for both experiment years, continue below on either screen
version_num =2
# Read in 1997/98 and 2015/16 dataframes
heat_budget_1997_df = pd.read_csv('/_data/heat_budget_enso_1997_v'+str(version_num)+'.csv', index_col=[0])
heat_budget_2015_df = pd.read_csv('/_data/heat_budget/heat_budget_enso_2015_v'+str(version_num)+'.csv', index_col=[0])
# Merge dataframes
heat_budget_df = pd.concat([heat_budget_1997_df,heat_budget_2015_df])

# Percent change in SHF between 2015/16 and 1997/98 event
((heat_budget_df[heat_budget_df['Year']==1997].SHF)-(heat_budget_df[heat_budget_df['Year']==2015].SHF))/heat_budget_df[heat_budget_df['Year']==2015].SHF

#%% PLOT SHF VS OHC
barWidth = 0.25
year_vec = [2015, 1997]
subplot_label=['A','B']
# Create plot
subplot_num=0
fig=plt.subplots(2,1,sharex=True,figsize=(6,8));
for yr in year_vec:
    plt.subplot(2,1,subplot_num+1);
    # Bar positions
    r = np.arange(len(heat_budget_df[heat_budget_df['Year']==yr]))
    r2 = r + barWidth
    plt.bar(r, heat_budget_df[heat_budget_df['Year']==yr]['SHF']/1e18, color='#bdc9e1', width=barWidth, edgecolor='white', label='Surface flux')
    plt.bar(r2, heat_budget_df[heat_budget_df['Year']==yr]['OHC']/1e18, color='#0570b0', width=barWidth, edgecolor='white', label='Ocean divergence (z<100 m)')
    # Set xticks
    plt.xticks(r + barWidth/2,labels=heat_budget_df[heat_budget_df['Year']==yr]['Experiment'])
    # Set ytick fontsize
    plt.yticks(fontsize=12);
    plt.ylim(-1.75,0.3);
    # Add horizontal dashed line
    plt.axhline(y=0, linestyle='--', color='k');
    
    # Set axis labels
    plt.ylabel('$\Delta$ Niño3.4 heat content (EJ)', fontsize=12);
    if subplot_num>0:
        plt.xlabel('MCB strategy',fontsize=12);
    elif subplot_num==0:
        # Add legend
        plt.legend(loc='lower right');
    # Add title
    plt.annotate(str(yr)+'-'+ str(int(float(yr)+1))+ ' El Niño',xy=(.5,1.08),xycoords='axes fraction',horizontalalignment='center', verticalalignment='top',fontsize=12);
    plt.title(subplot_label[subplot_num],fontsize=14,fontweight='bold',loc='left');
    subplot_num+=1
plt.tight_layout();
