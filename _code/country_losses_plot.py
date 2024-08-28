### PURPOSE: Plot country-level losses from MCB-modified El Niño
### AUTHOR: Jessica Wan (j4wan@ucsd.edu)
### DATE CREATED: 08/27/2024

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
from matplotlib import ticker
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.ion();

##################################################################################################################
## WHICH EXPERIMENT ARE YOU READING IN? ##
# month_init = input('Which initialization month are you reading in (02, 05, 08, 11)?: ')
# year_init = input('Which initialization year are you reading in (1997, 2015?): ')
# enso_phase = input('Which ENSO event are you reading in (nino or nina)?: ')
# sensitivity_opt = input('Sensitivity run (y or n)?: ')
# Hard code for 2015 testing
month_init = '05'
# year_init = '2015'
enso_phase = 'nino'
sensitivity_opt = 'y'
## GDP option (gross or pc)
gdp_opt = input('Gross (1) or per capita (2)?: ')
##################################################################################################################

#%% READ IN DATA
# Read in country geometry file
countries = gpd.read_file('/home/j4wan/Migration/projections/country_shp/ne_50m_admin_0_countries.shp')
countries = countries.rename(columns={'ISO_N3':'country_id'})
# Norway (ISO_N3=-99, need to manually add to dataframe)
countries.loc[88,'ISO_A3']='NOR'
countries.loc[88,'country_id']='578'
# Remove -99 country_id (5 rows)
countries = countries.drop(countries[countries['country_id']=='-99'].index)
countries['CountryID'] = countries.country_id.astype(int).astype(str).str.zfill(3).astype(float)

## Read in country-level economic losses from CM replication analysis
if gdp_opt=='1':
    loss = pd.read_csv('/_data/SMYLE-MCB/processed_data/callahan_regression/MCBbenefits_v2.csv',header=2)
elif gdp_opt=='2':
    loss = pd.read_csv('/_data/SMYLE-MCB/processed_data/callahan_regression/MCBbenefits_percapita.csv',header=2, na_values= '#DIV/0!')


# Subset columns
icols = [1,2,16,3,17,10,24,8,22,15,29]
icols.sort()
loss_subset = loss.iloc[:, icols]
# Rename columns
loss_clean = loss_subset.rename(columns={loss_subset.columns[0]:'ISO_A3',loss_subset.columns[1]:'Control_e_97',loss_subset.columns[2]:'06-02_e_97',loss_subset.columns[3]:'12-02_e_97',\
                                         loss_subset.columns[4]:'06-02_a_97',loss_subset.columns[5]:'12-02_a_97',\
                                         loss_subset.columns[6]:'Control_e_15',loss_subset.columns[7]:'06-02_e_15',loss_subset.columns[8]:'12-02_e_15',\
                                         loss_subset.columns[9]:'06-02_a_15',loss_subset.columns[10]:'12-02_a_15'})
 
## Merge country geometry and losses by ISO number
loss_merged = countries[['ISO_A3','geometry']].merge(loss_clean,on='ISO_A3')


if gdp_opt=='1':
    ## PLOT A COUNTRY-LEVEL LOSSES ($ total)
    version_num = 1
    fmt1 = ticker.ScalarFormatter(useMathText=True)
    fmt1.set_powerlimits((0, 0))
    nrow=5
    ncol=2
    fig = plt.figure(figsize=(6,8));
    subplot_num=0
    # a) 1997-98 Control
    vmin=-2e2
    vmax=2e2
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    p=loss_merged.plot(loss_merged.Control_e_97.values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('a', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.15,1.15,'1997-1998 El Niño', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # b) 2015-16 Control
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged.Control_e_15.values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('b', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.15,1.15,'2015-2016 El Niño', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # c) 1997-98 Full effort original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_e_97'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('c', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # d) 2015-16 Full effort original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_e_15'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('d', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # e) 1997-98 Full effort anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_a_97'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('e', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # f) 2015-16 Full effort anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_a_15'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('f', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # g) 1997-98 11th hour original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_e_97'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('g', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # h) 2015-16 11th hour original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_e_15'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('h', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # i) 1997-98 11th hour anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_a_97'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('i', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # j) 2015-1611th hour anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_a_15'].values/1e9,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP (billions of USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('j', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    # Add vertical text
    plt.figtext(0.05,0.85,'Control',fontsize=12,rotation=90);
    plt.figtext(0.05,0.52,'Full effort MCB',fontsize=12,rotation=90);
    plt.figtext(0.05,0.18,'11th hour MCB',fontsize=12,rotation=90);
    # Adjust subplot spacing
    fig.subplots_adjust(top=0.97,bottom = 0.1,hspace=.1);
    # Add colorbar
    mappable = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin, vmax),
        cmap='PRGn')
    # define position and extent of colorbar
    cbar_ax = fig.add_axes([0.115, 0.08, 0.8, 0.02]) #rect kwargs [left, bottom, width, height];
    # draw colorbar
    cbar = fig.colorbar(mappable, cax=cbar_ax, extend='both',label='GDP (billions of USD)',orientation='horizontal')

elif gdp_opt=='2':
    ## PLOT A COUNTRY-LEVEL LOSSES ($ total)
    fmt1 = ticker.ScalarFormatter(useMathText=True)
    fmt1.set_powerlimits((0, 0))
    nrow=5
    ncol=2
    fig = plt.figure(figsize=(6,8));
    subplot_num=0
    # a) 1997-98 Control
    vmin=-2e3
    vmax=2e3
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    p=loss_merged.plot(loss_merged.Control_e_97.values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('a', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.15,1.15,'1997-1998 El Niño', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # b) 2015-16 Controlf
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged.Control_e_15.values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('b', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.15,1.15,'2015-2016 El Niño', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # c) 1997-98 Full effort original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_e_97'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('c', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # d) 2015-16 Full effort original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_e_15'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('d', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # e) 1997-98 Full effort anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_a_97'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('e', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # f) 2015-16 Full effort anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['06-02_a_15'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('f', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # g) 1997-98 11th hour original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_e_97'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('g', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # h) 2015-16 11th hour original method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_e_15'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('h', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.35,1.15,'E-index', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # i) 1997-98 11th hour anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_a_97'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('i', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    subplot_num+=1
    # j) 2015-1611th hour anomaly method
    ax0 = plt.subplot(nrow,ncol,int(1+subplot_num), projection=ccrs.PlateCarree(),transform=ccrs.PlateCarree())
    loss_merged.plot(loss_merged['12-02_a_15'].values,legend=False,cmap='PRGn', vmin=vmin,vmax=vmax,ax=ax0,\
        legend_kwds={'pad':0.04,'orientation' : 'horizontal','label':'GDP per capita (USD)','extend':'both','format':fmt1,'fraction':0.09});
    ylims=ax0.get_ylim();
    loss_merged.boundary.plot(ax=ax0,color = 'lightgrey', lw = .2 );
    ax0.set_xlim((-180,180));ax0.set_ylim(-75,90);
    plt.tick_params(left=False,bottom=False);
    plt.xticks([]);plt.yticks([]);
    plt.title('j', fontweight='bold',loc='left',fontsize=12);
    ax0.text(0.32,1.15,'Anomaly', transform=ax0.transAxes,fontsize=12);
    # Add vertical text
    plt.figtext(0.05,0.85,'Control',fontsize=12,rotation=90);
    plt.figtext(0.05,0.52,'Full effort MCB',fontsize=12,rotation=90);
    plt.figtext(0.05,0.18,'11th hour MCB',fontsize=12,rotation=90);
    # Adjust subplot spacing
    fig.subplots_adjust(top=0.97,bottom = 0.1,hspace=.1);
    # Add colorbar
    mappable = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin, vmax),
        cmap='PRGn')
    # define position and extent of colorbar
    cbar_ax = fig.add_axes([0.115, 0.08, 0.8, 0.02]) #rect kwargs [left, bottom, width, height];
    # draw colorbar
    cbar = fig.colorbar(mappable, cax=cbar_ax, extend='both',label='GDP per capita (USD)',orientation='horizontal')


