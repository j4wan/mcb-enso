### PURPOSE: Script to store functions used in all analysis scripts in _code/
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 05/28/2024


### TABLE OF CONTENTS ### -------------------------------------------------------------
# 1) reorient_netCDF : orient and save netcdf wrt -180,180 longitude OR 0-360 longitude
# 2) globalarea: computes grid cell area for globe
# 3) dateshift_netCDF: shift and save netcdf with dates at midpoint of month
# 4) plot_panel_maps: plot maps of global/regional monthly means/anomalies by subplot panel
# 5) calc_weighted_mean_sd: calculate area weighted mean and standard deviation
# 6) calc_weighted_mean_tseries: calculate area weighted mean for time series
# 7) weighted_temporal_mean: calculate day-weighted mean
# 8) weighted_temporal_mean: calculate day-weighted mean
# 9) plot_nino_maps: plot surface temperature and ocean potential temperature cross-section
# 10) moving_average: calculate moving average
# 11) add_contourf3d: create contourf 3d plot
# 12) add_contour3d: create contour 3d plot
# 13) add_feature3d: Add the given feature to the given axes.
### ------------------------------------------------------------------------------- ###


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
import glob
import datetime
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
from matplotlib import ticker
from cartopy.mpl.patch import geos_to_path
import itertools
import mpl_toolkits.mplot3d
from matplotlib.collections import PolyCollection, LineCollection



# 1) reorient_netCDF : orient and save netcdf wrt -180,180 OR 0-360 longitude
def reorient_netCDF(fp,target=180):
    """
    Function to orient and save netcdf wrt -180,180 OR 0-360 longitude.
    :param fp: dataarray to be reoriented
    :param target: int (180 or 360) speecifying which way to orient longitude
    """
    f = fp
    if target==180:
        if np.max(f.coords['lon'])>180:
            new_lon = [-360.00 + num if num > 180 else num for num in f.coords['lon'].values]
            f = f.assign_coords({'lon':new_lon})
            f.assign_coords(lon=(np.mod(f.lon + 180, 360) - 180))
            f = f.sortby(f.coords['lon'])
    elif target==360:
        new_lon = [360.00 + num if num < 0 else num for num in f.coords['lon'].values]
        f = f.assign_coords({'lon':new_lon})
        f = f.sortby(f.coords['lon'])
    return f


# 2) globalarea: computes grid cell area for globe
def globalarea(xr_in):
    """
    globalarea(xr_in) gives the area [km^2] for each grid point of a regular global grid.
    default is to have lat/lon as 360 and 720, which is a 0.5 x 0.5 degree grid
    """
    lat_len = len(xr_in.lat)
    lon_len = len(xr_in.lon)
    dims=[lat_len,lon_len]
    theta = np.linspace(0,np.pi,dims[0]+1) # Define regular angle for latitude
    theta1 = theta[:-1]
    theta2 = theta[1:]
    Re = 6371. # radius of Earth (in km)
    dA = 2.*Re*Re*np.pi/dims[1]*(-np.cos(theta2)+np.cos(theta1))
    dAmat = np.tile(dA,(dims[1],1)).T # Repeat for each longitude
    return dAmat

# 3) dateshift_netCDF : shift and save netcdf with dates at midpoint of month
def dateshift_netCDF(fp):
    """
    Function to shift and save netcdf with dates at midpoint of month.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if np.unique(fp.indexes['time'].day)[0]==1 & len(np.unique(fp.indexes['time'].day))==1:
        new_time = fp.indexes['time']-datetime.timedelta(days=16)
        f = f.assign_coords({'time':new_time})
    return f


# 4) plot_panel_maps: plot maps of global/regional monthly means/anomalies by subplot panel
def plot_panel_maps(in_xr,cmin, cmax, ccmap, plot_zoom,central_lon=0,projection = 'PlateCarree',CI_in='none',CI_level='none',CI_display='none', nrow=1, ncol=1, subplot_num=0, mean_val='none',cbar=True):
    """
    Function to plot maps of global/regional ensemble mean and anomalies
    :param in_xr: xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and anomaly plots
    :param plot_zoom: string to specify zoom of plot ('global','conus', 'west_coast', 'pacific_ocean')
    :param central_lon: float specifying central longitude for plotting (default=0) If central_lon=180, need to add cyclical point to remove white line
    :param projection: string specifying plot projection. Regional facets only work for PlateCarree. Default is PlateCarree.
:param CI_in: xarray w/ same dims as in_xr specifying 1's where the grid cells are significant to the CI and 0's elsewhere. Default is None.
    :param CI_level: float specifying signficance level for plotting.
    :param CI_display: string specifying how to show CI (default='none'). Options include stippling significant pixels, inverted stippling where insignficant pixels are stippled, or masking out insignificant pixels.
    :param nrow: int specifying number of rows for subplot.
    :param ncol: int specifying number of cols for subplot.
    :param subplot_num: int specifying which subplot panel you are plotting.
    :param mean_val: default is none. If not, specify array(mean, std) to put mean value in top right corner.
    """

    x = in_xr
    x_ci = CI_in
    if projection=='PlateCarree':
        plot_proj = ccrs.PlateCarree(central_longitude=central_lon)
    elif projection=='Robinson':
        plot_proj = ccrs.Robinson(central_longitude=central_lon)
    elif projection=='Mollweide':
        plot_proj = ccrs.Mollweide(central_longitude=central_lon)
    elif projection=='LambertConformal':
        plot_proj = ccrs.LambertConformal(central_longitude=central_lon)
    ax = plt.subplot(nrow,ncol,int(1+subplot_num), projection=plot_proj,transform=plot_proj)

    # Remove white line if plotting over Pacific Ocean.
    if central_lon==180:
        lat = x.lat
        lon = x.lon
        data = x
        data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
        lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)

        # fig = plt.figure(figsize=(8,7));
        # ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)

        if plot_zoom=='global':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            p.axes.set_global();
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        elif plot_zoom=='conus':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150+180),(-65+180));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='west_coast':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140+180),(-100+180));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='pacific_ocean':
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            #ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
            if projection=='LambertConformal':
                ax.set_extent([-270, -90, 15, 80], ccrs.PlateCarree());
            else:
                ax.set_extent([-270, -90, 0, 80], ccrs.PlateCarree());
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--');
            gl.xlines=False;gl.ylines=False;
            gl.ylocator = ticker.FixedLocator([0, 30, 60])
            gl.xlocator = ticker.FixedLocator([-120, 180, 120])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            # ax.yaxis.set_major_formatter(LatitudeFormatter());ax.xaxis.set_major_formatter(LongitudeFormatter());
            gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True                    
        ax.coastlines()
        # fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
        #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
        if cbar==True:
            plt.colorbar(p,orientation='horizontal', label=x.units, extend='both',pad=0.1);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=10, loc = 'right');
    # Central longitude is default 0.
    else:
        lat = x.lat
        lon = x.lon
        data = x
        lat_mesh, lon_mesh  = np.meshgrid(lon,lat)

        # fig = plt.figure(figsize=(8,6));
        # ax = plt.subplot(1,1,1, projection=plot_proj,transform=plot_proj)

        if plot_zoom=='global':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]<=CI_level],lon_mesh[::4,::6][x_ci[::4,::6]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[::4,::6][x_ci[::4,::6]>CI_level],lon_mesh[::4,::6][x_ci[::4,::6]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            p.axes.set_global();
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True
        elif plot_zoom=='conus':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                 
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(15,60);ax.set_xlim((-150),(-65));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='west_coast':
            if CI_display == 'mask':
                data = xr.where(x_ci<=CI_level,data,np.nan)
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                ax.scatter(lat_mesh[x_ci<=CI_level],lon_mesh[x_ci<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon, lat, data,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())                     
                ax.scatter(lat_mesh[x_ci>CI_level],lon_mesh[x_ci>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(30,55);ax.set_xlim((-140),(-100));
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False
        elif plot_zoom=='pacific_ocean':
            lat = x.lat
            lon = x.lon
            data = x
            data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
            lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)
            if CI_display == 'mask':
                ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                data_wrap = xr.where(ci_wrap<=CI_level,data_wrap,np.nan)
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
            elif CI_display == 'stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]<=CI_level],lon_mesh[::2,::3][x_ci[::2,::3]<=CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            elif CI_display == 'inv_stipple':
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
                #ci_wrap, lon_wrap = add_cyclic_point(x_ci,coord=lon)
                lat_mesh, lon_mesh  = np.meshgrid(lon,lat)                        
                ax.scatter(lat_mesh[::2,::3][x_ci[::2,::3]>CI_level],lon_mesh[::2,::3][x_ci[::2,::3]>CI_level], transform= ccrs.PlateCarree(), color='k',s=0.25,alpha=0.6)
            else:
                p = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())    
            ax.add_feature(cfeature.STATES, edgecolor='k',linewidth=0.5);ax.set_ylim(0,80);ax.set_xlim(-80,80);
            #ax.set_yticks([0, 20, 40, 60, 80], crs=plot_proj)
            #ax.yaxis.set_major_formatter(LatitudeFormatter())
            #gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=False                    
        ax.coastlines()
        #fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
        #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
        if cbar==True:
            plt.colorbar(p,orientation='horizontal', label=x.units, extend='both',pad=0.1);
        if mean_val!='none':
            plt.title(str(round(mean_val[0],2))+ ' '+str(x.units), fontsize=10, loc = 'right');
    return ax, p



# 5) calc_weighted_mean_sd: calculate area weighted mean and standard deviation
def calc_weighted_mean_sd(DataArray):
    '''
    Calculate area-weighted aggregate mean of a variable in an input DataArray
    Adapted from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    Returns a two values: the area-weighted mean over time of the variable and standard deviation of the means
        over whatever spatial domain is specified in the input DataArray
    '''
    # create array of weights for each grid cell based on latitude
    weights = np.cos(np.deg2rad(DataArray.lat))
    weights.name = "weights"
    array_weighted = DataArray.weighted(weights)
    weighted_mean = array_weighted.mean(("lon", "lat"))
    aggregate_weighted_sd = weighted_mean.std(dim='time')
    aggregate_weighted_mean = weighted_mean.mean(dim='time')
    return float(aggregate_weighted_mean.values), float(aggregate_weighted_sd.values)

# 6) calc_weighted_mean_tseries: calculate area weighted mean for time series
def calc_weighted_mean_tseries(DataArray):
    '''
    Calculate area-weighted aggregate mean of a variable in an input DataArray
    Adapted from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    Returns an array: the area-weighted mean over time of the variable
        over whatever spatial domain is specified in the input DataArray
    '''
    # create array of weights for each grid cell based on latitude
    weights = np.cos(np.deg2rad(DataArray.lat))
    weights.name = "weights"
    array_weighted = DataArray.weighted(weights)
    weighted_mean = array_weighted.mean(("lon", "lat"))
    return weighted_mean


# 7) weighted_temporal_mean: calculate day-weighted mean
def weighted_temporal_mean(ds):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds * 1.0

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="AS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="AS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out


# 8) weighted_temporal_mean: calculate day-weighted mean
def weighted_temporal_mean_clim(ds):
    """
    weight by days in each month for monthly climatology
    """
    # Determine the month length
    month_dict = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    month_length = xr.DataArray(data=np.array([month_dict.get(key) for key in ds.month.values]),dims='month',coords=dict(month=ds.month))

    # Calculate the weights
    wgts = month_length / month_length.sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.sum(), 1.0)

    # Subset our dataset for our variable
    obs = ds * 1.0

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).sum(dim='month')

    # Calculate the denominator
    ones_out = (ones * wgts).sum(dim='month')

    # Return the weighted average
    return obs_sum / ones_out



# 9) plot_nino_maps: plot surface temperature and ocean potential temperature cross-section
def plot_nino_maps(atm_in_xr, ocn_in_xr, cmin, cmax, ccmap, u_in_xr='none', v_in_xr='none', u_scale=10):
    """
    Function to plot maps of global/regional ensemble mean and anomalies
    :param atm_in_xr: xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param u_in_xr: optional xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param v_in_xr: optional xarray w/ dims [lat,lon] representing annual mean/anomaly
    :param u_scale: optional integer for scaling wind vector
    :param ocn_in_xr: xarray w/ dims [z_t,lon] representing annual mean/anomaly
    :param cmin: float minimum value for ensemble mean and seasonal climatology plots
    :param cmax: float maximum value for ensemble mean and seasonal climatology plots
    :param ccmap: string colormap pallette for ensemble mean and anomaly plots
    """

    ## SURFACE TEMPERATURE
    x = atm_in_xr
    plot_proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.subplot(2,1,1, projection=plot_proj,transform=plot_proj)

    # Remove white line if plotting over Pacific Ocean.
    lat = x.lat
    lon = x.lon
    data = x
    data_wrap, lon_wrap = add_cyclic_point(data,coord=lon)
    lat_mesh, lon_mesh  = np.meshgrid(lon_wrap,lat)


    p1 = ax.pcolormesh(lon_wrap, lat, data_wrap,vmin=cmin,vmax=cmax,cmap=ccmap,transform= ccrs.PlateCarree())
    if str(type(u_in_xr))=="<class 'xarray.core.dataarray.DataArray'>":
        m1 = plt.quiver(u_in_xr.lon.values[::10], u_in_xr.lat.values[::10],\
                u_in_xr.values[::10,::10],v_in_xr.values[::10,::10],\
                transform=ccrs.PlateCarree(), units='width', pivot='middle', color='k',width=0.0025);
        plt.quiverkey(m1, X=6, Y=3, U=u_scale, label=str(u_scale)+' ms$^{-1}$', labelpos='E', coordinates = 'inches');    
    ax.set_extent([-230, -75, -25, 25], ccrs.PlateCarree());
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=1, color='gray', alpha=0.5, linestyle='--');
    gl.xlines=True;gl.ylines=True;
    gl.ylocator = ticker.FixedLocator([-15, 0, 15])
    gl.xlocator = ticker.FixedLocator([-90, -120, -150, 180, 150])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # ax.yaxis.set_major_formatter(LatitudeFormatter());ax.xaxis.set_major_formatter(LongitudeFormatter());
    gl.top_labels=False;gl.bottom_labels=True; gl.right_labels=False; gl.left_labels=True                    
    ax.coastlines()
    plt.title('a',fontsize=12,fontweight='bold',loc='left');
     # fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.1,hspace=.01);
    #cbar_ax = fig.add_axes([0.115, 0.15, 0.8, 0.05]) #rect kwargs [left, bottom, width, height];
    #plt.colorbar(p,orientation='horizontal', label=x.units, extend='both',pad=0.1);
    
    ## POTENTIAL TEMPERATURE
    x = ocn_in_xr
    ax = plt.subplot(2,1,2)

    # Create mesh 
    z_t = x.z_t
    lon = x.lon
    data = x
    z_t_mesh, lon_mesh = np.meshgrid(x.lon, x.z_t)

    p2 = ax.pcolormesh(z_t_mesh, lon_mesh, x,vmin=cmin,vmax=cmax,cmap=ccmap)    
    plt.ylim(300,0);plt.xlim(135,280);
    ax.set_xticks([150,180,210,240,270])
    ax.set_xticklabels(['150°E','180°E', '150°W', '120°W','90°W'])
    CS = ax.contour(z_t_mesh, lon_mesh, x, [20], linewidths = 1, colors='k');
    ax.clabel(CS, inline=True, fontsize=8);
    #plt.colorbar(orientation='horizontal',label='Sv',extend='both');
    # plt.xlabel('Latitude');
    plt.ylabel('Depth ('+z_t.units+')',fontsize=12);
    plt.title('b',fontsize=12,fontweight='bold',loc='left');
    plt.colorbar(p1,orientation='horizontal', label=x.units, extend='both',pad=0.1);

    return ax


# 10) moving_average: calculate moving average
def moving_average(a, n) :
    """
    Function to calculate moving average over specified monthly range.
    :param a: array over which you want to calculate moving average.
    :param n: int specifying number of months over which you want to calculate average.

    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 11) add_contourf3d: create contourf 3d plot
def add_contourf3d(ax, contour_set):
    """
    Function to create contourf 3d plot. 
    Adapted from https://stackoverflow.com/questions/48269014/contourf-in-3d-cartopy
    """
    proj_ax = contour_set.collections[0].axes
    for zlev, collection in zip(contour_set.levels, contour_set.collections):
        paths = collection.get_paths()
        # Figure out the matplotlib transform to take us from the X, Y
        # coordinates to the projection coordinates.
        trans_to_proj = collection.get_transform() - proj_ax.transData

        paths = [trans_to_proj.transform_path(path) for path in paths]
        verts = [path.vertices for path in paths]
        codes = [path.codes for path in paths]
        pc = PolyCollection([])
        pc.set_verts_and_codes(verts, codes)

        # Copy all of the parameters from the contour (like colors) manually.
        # Ideally we would use update_from, but that also copies things like
        # the transform, and messes up the 3d plot.
        pc.set_facecolor(collection.get_facecolor())
        pc.set_edgecolor(collection.get_edgecolor())
        pc.set_alpha(collection.get_alpha())

        ax3d.add_collection3d(pc, zs=0)

    # Update the limit of the 3d axes based on the limit of the axes that
    # produced the contour.
    proj_ax.autoscale_view()

    ax3d.set_xlim(*proj_ax.get_xlim())
    ax3d.set_ylim(*proj_ax.get_ylim())
    # ax3d.set_zlim(Z.min(), Z.max())


# 12) add_contour3d: create contour 3d plot
def add_contour3d(ax, contour_set):
    """
    Function to create contour 3d plot. 
    Adapted from https://stackoverflow.com/questions/48269014/contourf-in-3d-cartopy
    """
    proj_ax = contour_set.collections[0].axes
    for zlev, collection in zip(contour_set.levels, contour_set.collections):
        paths = collection.get_paths()
        # Figure out the matplotlib transform to take us from the X, Y
        # coordinates to the projection coordinates.
        trans_to_proj = collection.get_transform() - proj_ax.transData

        paths = [trans_to_proj.transform_path(path) for path in paths]
        verts = [path.vertices for path in paths]
        codes = [path.codes for path in paths]
        pc = PolyCollection([])
        pc.set_verts_and_codes(verts, codes)

        # Copy all of the parameters from the contour (like colors) manually.
        # Ideally we would use update_from, but that also copies things like
        # the transform, and messes up the 3d plot.
        pc.set_facecolor((1,1,1,0))
        pc.set_edgecolor(collection.get_edgecolor())
        pc.set_alpha(collection.get_alpha())
        pc.set_linestyle(collection.get_linestyle())
        pc.set_linewidth(collection.get_linewidth())

        ax3d.add_collection3d(pc, zs=0)

    # Update the limit of the 3d axes based on the limit of the axes that
    # produced the contour.
    proj_ax.autoscale_view()

    ax3d.set_xlim(*proj_ax.get_xlim())
    ax3d.set_ylim(*proj_ax.get_ylim())
    # ax3d.set_zlim(Z.min(), Z.max())

# 13) add_feature3d: Add the given feature to the given axes.
def add_feature3d(ax, feature, clip_geom=None, zs=None):
    """
    Function to add the given feature to the given axes.
    Adapted from https://stackoverflow.com/questions/48269014/contourf-in-3d-cartopy
    """
    concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

    target_projection = ax.projection
    geoms = list(feature.geometries())

    if target_projection != feature.crs:
        # Transform the geometries from the feature's CRS into the
        # desired projection.
        geoms = [target_projection.project_geometry(geom, feature.crs)
                 for geom in geoms]

    if clip_geom:
        # Clip the geometries based on the extent of the map (because mpl3d
        # can't do it for us)
        geoms = [geom.intersection(clip_geom) for geom in geoms]

    # Convert the geometries to paths so we can use them in matplotlib.
    paths = concat(geos_to_path(geom) for geom in geoms)

    # Bug: mpl3d can't handle edgecolor='face'
    kwargs = feature.kwargs
    if kwargs.get('edgecolor') == 'face':
        kwargs['edgecolor'] = 'k'
        kwargs['facecolor'] = 'grey'
    polys = concat(path.to_polygons(closed_only=False) for path in paths)

    if kwargs.get('facecolor', 'none') == 'none':
        lc = LineCollection(polys, **kwargs)
    else:
        lc = PolyCollection(polys, closed=False,**kwargs)
    ax3d.add_collection3d(lc, zs=zs)
