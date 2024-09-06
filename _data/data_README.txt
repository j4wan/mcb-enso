Data README
You will need to populate _data/ with associated data files available from NCAR’s campaign storage system (accessible from Casper).
MCB experiments:
/glade/campaign/cesm/development/espwg/SMYLE-MCB/timeseries
SMYLE-AUFIRE experiments:
/glade/campaign/cesm/development/espwg/SMYLE-AUFIRE/archive
SMYLE control and historical experiments:
/glade/campaign/cesm/development/espwg/SMYLE/archive

Sample data files to demo the code have been provided. For CESM2 output, select variables from 1 ensemble member have been uploaded. Note that due to storage limitations, no ocean output, wind (U and V) variables, historical climatologies, or georeferenced country level  anomalies have been provided but they are available upon request (Correspondence to Jessica Wan; j4wan@ucsd.edu).

Contents:
sesp_mask_CESM2_0.9x1.25_v3.nc : example MCB seeding mask.
/callahan_regression:
  BenefitEstimateENSOMCB_codeavailability.xlsx : C&M replication analysis (see README in file for more details)
  intermediate csvs produced from ../_code for C&M replication analysis
/country_shp: 
  ne_50m_admin_0_countries.shp: country shapefile for plotting and country aggregation.
/enso_regions: 
  djf_major_nino_regions_05_0.1_sigma_v2.nc: netcdf defining 0.1 sigma wet/dry and cool/warm regions for major El Niño events from SMYLE   output.
/LENS2:
  CESM2 LENS2 historical ensemble standard deviation netcdfs for surface temperature (*TS*) and precipitation (*PRECT*) from 1970-2014.
/MCB:
  /b.e21.BSMYLE.f09_g17.MCB_2015-05_06-02.2015-05.101: CESM2 atm output for 1 ensemble member of Full effort MCB during 2015/16 El Niño.
  /b.e21.BSMYLE.f09_g17.MCB.2019-08.001: CESM2 atm output for 1 ensemble member of MCB during 2019/20 La Niña.
/pop_data:
  netcdf and csv of population count data for population weighting from GPW v4.
/realtime:
  /b.e21.BSMYLE.f09_g17.2015-05.001: CESM2 atm output for 1 ensemble member of no MCB (control) during 2015/16 El Niño.
/SMYLE-AUFIRE:
  /b.e21.BSMYLE-AUFIRE.f09_g17.2019-08.201: CESM2 atm output for 1 ensemble member of the AUS wildfires during the 2019/20 La Niña (see     Yeager et al., 2022)
  
