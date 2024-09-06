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
/callahan_regression:
  BenefitEstimateENSOMCB_codeavailability.xlsx : C&M replication analysis (see README in file for more details)
  intermediate csvs produced from ../_code for C&M replication
sesp_mask_CESM2_0.9x1.25_v3.nc : example MCB seeding mask.
