### PURPOSE: Script to create SMYLE drift climatologies (n month 1 to 24)
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
plt.ion();

dask.config.set({"array.slicing.split_large_chunks": False})

##################################################################################################################
## THIS SCRIPT READS IN ONE ENSEMBLE OF EXPERIMENTS AT A TIME. ##
## WHICH EXPERIMENT ARE YOU READING IN? ##
month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
##################################################################################################################

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
clim_vars = ['SWCF','TS','PRECT','U','V','PS','CLDLOW','LHFLX','CLDLIQ','AWNC','FREQL']

# Conversion constants
# PRECT
m_to_mm = 1e3 #mm/m
s_to_days = 86400 #s/day

## READ IN CONTROL SMYLE HISTORICAL SIMULATIONS
# Read in each ensemble member as a discontinuous time series by concatenating overlapping periods
# Set target directory for processed files (this is just a placeholder)
target_dir='/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/processed'
version_num=3

# Make the target directory if necessary
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
# Process each ensemble member and save as a concatenated file with ensemble member as a dimension
atm_monthly_ctrl_clim = {}
for m in clim_members:
    print(m)
    combined_vars=xr.Dataset()
    for var in clim_vars:
        print(var)
        file_subset_clim =  sorted(glob.glob('/_data/SMYLE_clim/BSMYLE.1970-2019-'+month_init+'/atm_tseries/'+var+'/b.e21.BSMYLE.f09_g17.*'+m+'.cam*'))
        for file in file_subset_clim:
            if file_subset_clim.index(file)==0:
                if (var=='U') or (var=='V'):
                    # Select near-surface winds only
                    da_merged = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var].isel(lev=-1)
                elif (var=='CLDLIQ') or (var=='AWNC') or (var=='FREQL'):
                    da_merged = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var]
                    da_merged = da_merged.where(da_merged.lev>850,drop=True)
                else:
                    da_merged = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var]
            else:
                if (var=='U') or (var=='V'):
                    # Select near-surface winds only
                    next_file = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var].isel(lev=-1)
                elif (var=='CLDLIQ') or (var=='AWNC') or (var=='FREQL'):
                    next_file = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var]
                    next_file = next_file.where(next_file.lev>850,drop=True)
                else:
                    next_file = fun.dateshift_netCDF(fun.reorient_netCDF(xr.open_dataset(file)))[var]
                da_merged = xr.concat([da_merged, next_file], dim='time')           
        combined_vars=xr.merge([combined_vars,da_merged])
    # Convert time to datetime index
    combined_vars = combined_vars.assign_coords(time=combined_vars.indexes['time'].to_datetimeindex())
    # Subset time from 1970-2018 and compute climatology before combining ensemble members
    combined_vars_subset = combined_vars.isel(time=combined_vars['time.year']<2019)
    # combined_vars_subset = combined_vars
    atm_monthly_ctrl_clim[m] = combined_vars_subset
# Combine all files into one xarray dataset with ensemble members as a new dimension
atm_monthly_ctrl_clim_xr = xr.concat(list(map(atm_monthly_ctrl_clim.get, clim_members)),pd.Index(clim_members,name='member'))
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
# Calculate BL CDNUMC
atm_monthly_ctrl_clim_xr = atm_monthly_ctrl_clim_xr.assign(CDNUMC=(atm_monthly_ctrl_clim_xr['AWNC']/(1e6)/atm_monthly_ctrl_clim_xr['FREQL']).mean(dim='lev'))
atm_monthly_ctrl_clim_xr['CDNUMC'].attrs['units'] = '#/cm3'
atm_monthly_ctrl_clim_xr['CDNUMC'].load()
# Calculate total BL CLDLIQ
atm_monthly_ctrl_clim_xr = atm_monthly_ctrl_clim_xr.assign(CLDLIQ_BL=atm_monthly_ctrl_clim_xr['CLDLIQ'].sum(dim='lev'))
atm_monthly_ctrl_clim_xr['CLDLIQ_BL'].attrs['units'] = atm_monthly_ctrl_clim_xr['CLDLIQ'].units
atm_monthly_ctrl_clim_xr['CLDLIQ_BL'].load()

# Set individual SMYLE run duration
smyle_length=24
# Set time step for climatology
nmonth=np.arange(0,smyle_length,1)
# Create empty dictionary to populate
atm_monthly_drift_clim = {}
# Loop through each month n of a SMYLE run
for n in nmonth:
    # Compute mean of every n month over hindcast
    nmean = atm_monthly_ctrl_clim_xr.isel(time=slice(n,None,smyle_length)).mean(dim='time')
    atm_monthly_drift_clim[n] = nmean
# Combine across n to form drift climatology
atm_monthly_drift_clim_xr = xr.concat(list(map(atm_monthly_drift_clim.get, nmonth)),pd.Index(nmonth,name='time'))
### EXPORT PROCESSED NETCDF
atm_monthly_drift_clim_xr.to_netcdf(target_dir+'/BSMYLE.'+str(pd.to_datetime(combined_vars_subset.time.values[0]).year)+'-'+str(pd.to_datetime(combined_vars_subset.time.values[-1]).year)+'-'+month_init+'.atm_drift_clim.v'+str(version_num)+'.nc',mode='w',format='NETCDF4')
