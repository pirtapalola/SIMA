# Package ID: knb-lter-mcr.5.25 Cataloging System:https://pasta.edirepository.org.
# Data set title: MCR LTER: Coral Reef: Optical parameters and SST from SeaWiFS and MODIS,
# ongoing since 1997 and AVHRR-derived SST from 1985 to 2009.
# Data set creator:    - Moorea Coral Reef LTER
# Data set creator:  Stephane Maritorena - Moorea Coral Reef LTER
# Contact:    - Information Manager Moorea Coral Reef LTER  - mcrlter@msi.ucsb.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu
#
# This program creates numbered PANDA dataframes named dt1,dt2,dt3...,
# one for each data table in the dataset. It also provides some basic
# summaries of their contents. NumPy and Pandas modules need to be installed
# for the program to run.

import numpy as np
import pandas as pd

infile1 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/5/25/dee78fa641cd7550765c248cc13bb010".strip()
infile1 = infile1.replace("https://", "http://")

dt1 = pd.read_csv(infile1
                  , skiprows=1
                  , sep=","
                  , quotechar='"'
                  , names=[
        "Year",
        "Month",
        "first_day",
        "last_day",
        "mean_chl_seawifs",
        "stdev_chl_seawifs",
        "chl_anomaly_seawifs",
        "pct_valid_chl_seawifs",
        "mean_chl_aqua",
        "stdev_chl_aqua",
        "chl_anomaly_aqua",
        "pct_valid_chl_aqua",
        "cdm_mean_seawifs",
        "cdm_stdev_seawifs",
        "cdm_anomaly_seawifs",
        "pct_valid_cdm_seawifs",
        "cdm_mean_aqua",
        "cdm_stdev_aqua",
        "cdm_anomaly_aqua",
        "pct_valid_cdm_aqua",
        "mean_bbp_aqua",
        "stdev_bbp_aqua",
        "bbp_anomaly_aqua",
        "pct_valid_bbp_aqua",
        "mean_nsst_aqua",
        "stdev_nsst_aqua",
        "nsst_anomaly_aqua",
        "pct_valid_nsst_aqua"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'Year':'str' ,
                  #             'Month':'str',
                  #             'first_day':'str',
                  #             'last_day':'str',
                  #             'mean_chl_seawifs':'float' ,
                  #             'stdev_chl_seawifs':'float' ,
                  #             'chl_anomaly_seawifs':'float' ,
                  #             'pct_valid_chl_seawifs':'float' ,
                  #             'mean_chl_aqua':'float' ,
                  #             'stdev_chl_aqua':'float' ,
                  #             'chl_anomaly_aqua':'float' ,
                  #             'pct_valid_chl_aqua':'float' ,
                  #             'cdm_mean_seawifs':'float' ,
                  #             'cdm_stdev_seawifs':'float' ,
                  #             'cdm_anomaly_seawifs':'float' ,
                  #             'pct_valid_cdm_seawifs':'float' ,
                  #             'cdm_mean_aqua':'float' ,
                  #             'cdm_stdev_aqua':'float' ,
                  #             'cdm_anomaly_aqua':'float' ,
                  #             'pct_valid_cdm_aqua':'float' ,
                  #             'mean_bbp_aqua':'float' ,
                  #             'stdev_bbp_aqua':'float' ,
                  #             'bbp_anomaly_aqua':'float' ,
                  #             'pct_valid_bbp_aqua':'float' ,
                  #             'mean_nsst_aqua':'float' ,
                  #             'stdev_nsst_aqua':'float' ,
                  #             'nsst_anomaly_aqua':'float' ,
                  #             'pct_valid_nsst_aqua':'float'
                  #        }
                  , parse_dates=[
        'Year',
    ]
                  , na_values={
        'mean_chl_seawifs': [
            '-999.000', ],
        'stdev_chl_seawifs': [
            '-999.000', ],
        'chl_anomaly_seawifs': [
            '-999.000', ],
        'pct_valid_chl_seawifs': [
            '-999.000', ],
        'mean_chl_aqua': [
            '-999.000', ],
        'stdev_chl_aqua': [
            '-999.000', ],
        'chl_anomaly_aqua': [
            '-999.000', ],
        'pct_valid_chl_aqua': [
            '-999.000', ],
        'cdm_mean_seawifs': [
            '-999.000', ],
        'cdm_stdev_seawifs': [
            '-999.000', ],
        'cdm_anomaly_seawifs': [
            '-999.000', ],
        'pct_valid_cdm_seawifs': [
            '-999.000', ],
        'cdm_mean_aqua': [
            '-999.000', ],
        'cdm_stdev_aqua': [
            '-999.000', ],
        'cdm_anomaly_aqua': [
            '-999.000', ],
        'pct_valid_cdm_aqua': [
            '-999.000', ],
        'mean_bbp_aqua': [
            '-999.000', ],
        'stdev_bbp_aqua': [
            '-999.000', ],
        'bbp_anomaly_aqua': [
            '-999.000', ],
        'pct_valid_bbp_aqua': [
            '-999.000', ],
        'mean_nsst_aqua': [
            '-999.000', ],
        'stdev_nsst_aqua': [
            '-999.000', ],
        'nsst_anomaly_aqua': [
            '-999', ],
        'pct_valid_nsst_aqua': [
            '-999.000', ], }

                  )
# Coerce the data into the types specified in the metadata
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt1 = dt1.assign(Year_datetime=pd.to_datetime(dt1.Year, errors='coerce'))
dt1.Month = str(dt1.Month)
dt1.first_day = str(dt1.first_day)
dt1.last_day = str(dt1.last_day)
dt1.mean_chl_seawifs = pd.to_numeric(dt1.mean_chl_seawifs, errors='coerce')
dt1.stdev_chl_seawifs = pd.to_numeric(dt1.stdev_chl_seawifs, errors='coerce')
dt1.chl_anomaly_seawifs = pd.to_numeric(dt1.chl_anomaly_seawifs, errors='coerce')
dt1.pct_valid_chl_seawifs = pd.to_numeric(dt1.pct_valid_chl_seawifs, errors='coerce')
dt1.mean_chl_aqua = pd.to_numeric(dt1.mean_chl_aqua, errors='coerce')
dt1.stdev_chl_aqua = pd.to_numeric(dt1.stdev_chl_aqua, errors='coerce')
dt1.chl_anomaly_aqua = pd.to_numeric(dt1.chl_anomaly_aqua, errors='coerce')
dt1.pct_valid_chl_aqua = pd.to_numeric(dt1.pct_valid_chl_aqua, errors='coerce')
dt1.cdm_mean_seawifs = pd.to_numeric(dt1.cdm_mean_seawifs, errors='coerce')
dt1.cdm_stdev_seawifs = pd.to_numeric(dt1.cdm_stdev_seawifs, errors='coerce')
dt1.cdm_anomaly_seawifs = pd.to_numeric(dt1.cdm_anomaly_seawifs, errors='coerce')
dt1.pct_valid_cdm_seawifs = pd.to_numeric(dt1.pct_valid_cdm_seawifs, errors='coerce')
dt1.cdm_mean_aqua = pd.to_numeric(dt1.cdm_mean_aqua, errors='coerce')
dt1.cdm_stdev_aqua = pd.to_numeric(dt1.cdm_stdev_aqua, errors='coerce')
dt1.cdm_anomaly_aqua = pd.to_numeric(dt1.cdm_anomaly_aqua, errors='coerce')
dt1.pct_valid_cdm_aqua = pd.to_numeric(dt1.pct_valid_cdm_aqua, errors='coerce')
dt1.mean_bbp_aqua = pd.to_numeric(dt1.mean_bbp_aqua, errors='coerce')
dt1.stdev_bbp_aqua = pd.to_numeric(dt1.stdev_bbp_aqua, errors='coerce')
dt1.bbp_anomaly_aqua = pd.to_numeric(dt1.bbp_anomaly_aqua, errors='coerce')
dt1.pct_valid_bbp_aqua = pd.to_numeric(dt1.pct_valid_bbp_aqua, errors='coerce')
dt1.mean_nsst_aqua = pd.to_numeric(dt1.mean_nsst_aqua, errors='coerce')
dt1.stdev_nsst_aqua = pd.to_numeric(dt1.stdev_nsst_aqua, errors='coerce')
dt1.nsst_anomaly_aqua = pd.to_numeric(dt1.nsst_anomaly_aqua, errors='coerce')
dt1.pct_valid_nsst_aqua = pd.to_numeric(dt1.pct_valid_nsst_aqua, errors='coerce')

print("Here is a description of the data frame dt1 and number of lines\n")
print(dt1.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt1\n")
print(dt1.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt1.Year.describe())
print("--------------------\n\n")

print(dt1.Month.describe())
print("--------------------\n\n")

print(dt1.first_day.describe())
print("--------------------\n\n")

print(dt1.last_day.describe())
print("--------------------\n\n")

print(dt1.mean_chl_seawifs.describe())
print("--------------------\n\n")

print(dt1.stdev_chl_seawifs.describe())
print("--------------------\n\n")

print(dt1.chl_anomaly_seawifs.describe())
print("--------------------\n\n")

print(dt1.pct_valid_chl_seawifs.describe())
print("--------------------\n\n")

print(dt1.mean_chl_aqua.describe())
print("--------------------\n\n")

print(dt1.stdev_chl_aqua.describe())
print("--------------------\n\n")

print(dt1.chl_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt1.pct_valid_chl_aqua.describe())
print("--------------------\n\n")

print(dt1.cdm_mean_seawifs.describe())
print("--------------------\n\n")

print(dt1.cdm_stdev_seawifs.describe())
print("--------------------\n\n")

print(dt1.cdm_anomaly_seawifs.describe())
print("--------------------\n\n")

print(dt1.pct_valid_cdm_seawifs.describe())
print("--------------------\n\n")

print(dt1.cdm_mean_aqua.describe())
print("--------------------\n\n")

print(dt1.cdm_stdev_aqua.describe())
print("--------------------\n\n")

print(dt1.cdm_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt1.pct_valid_cdm_aqua.describe())
print("--------------------\n\n")

print(dt1.mean_bbp_aqua.describe())
print("--------------------\n\n")

print(dt1.stdev_bbp_aqua.describe())
print("--------------------\n\n")

print(dt1.bbp_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt1.pct_valid_bbp_aqua.describe())
print("--------------------\n\n")

print(dt1.mean_nsst_aqua.describe())
print("--------------------\n\n")

print(dt1.stdev_nsst_aqua.describe())
print("--------------------\n\n")

print(dt1.nsst_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt1.pct_valid_nsst_aqua.describe())
print("--------------------\n\n")

infile2 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/5/25/aa1a71b7cb1629d885fc1129e72553bb".strip()
infile2 = infile2.replace("https://", "http://")

dt2 = pd.read_csv(infile2
                  , skiprows=1
                  , sep=","
                  , quotechar='"'
                  , names=[
        "Year",
        "Month",
        "first_day",
        "last_day",
        "mean_chl_seawifs",
        "stdev_chl_seawifs",
        "chl_anomaly_seawifs",
        "pct_valid_chl_seawifs",
        "mean_chl_aqua",
        "stdev_chl_aqua",
        "chl_anomaly_aqua",
        "pct_valid_chl_aqua",
        "cdm_mean_seawifs",
        "cdm_stdev_seawifs",
        "cdm_anomaly_seawifs",
        "pct_valid_cdm_seawifs",
        "cdm_mean_aqua",
        "cdm_stdev_aqua",
        "cdm_anomaly_aqua",
        "pct_valid_cdm_aqua",
        "mean_bbp_aqua",
        "stdev_bbp_aqua",
        "bbp_anomaly_aqua",
        "pct_valid_bbp_aqua",
        "mean_nsst_aqua",
        "stdev_nsst_aqua",
        "nsst_anomaly_aqua",
        "pct_valid_nsst_aqua"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'Year':'str' ,
                  #             'Month':'str',
                  #             'first_day':'str',
                  #             'last_day':'str',
                  #             'mean_chl_seawifs':'float' ,
                  #             'stdev_chl_seawifs':'float' ,
                  #             'chl_anomaly_seawifs':'float' ,
                  #             'pct_valid_chl_seawifs':'float' ,
                  #             'mean_chl_aqua':'float' ,
                  #             'stdev_chl_aqua':'float' ,
                  #             'chl_anomaly_aqua':'float' ,
                  #             'pct_valid_chl_aqua':'float' ,
                  #             'cdm_mean_seawifs':'float' ,
                  #             'cdm_stdev_seawifs':'float' ,
                  #             'cdm_anomaly_seawifs':'float' ,
                  #             'pct_valid_cdm_seawifs':'float' ,
                  #             'cdm_mean_aqua':'float' ,
                  #             'cdm_stdev_aqua':'float' ,
                  #             'cdm_anomaly_aqua':'float' ,
                  #             'pct_valid_cdm_aqua':'float' ,
                  #             'mean_bbp_aqua':'float' ,
                  #             'stdev_bbp_aqua':'float' ,
                  #             'bbp_anomaly_aqua':'float' ,
                  #             'pct_valid_bbp_aqua':'float' ,
                  #             'mean_nsst_aqua':'float' ,
                  #             'stdev_nsst_aqua':'float' ,
                  #             'nsst_anomaly_aqua':'float' ,
                  #             'pct_valid_nsst_aqua':'float'
                  #        }
                  , parse_dates=[
        'Year',
    ]
                  , na_values={
        'mean_chl_seawifs': [
            '-999.000', ],
        'stdev_chl_seawifs': [
            '-999.000', ],
        'chl_anomaly_seawifs': [
            '-999.000', ],
        'pct_valid_chl_seawifs': [
            '-999.000', ],
        'mean_chl_aqua': [
            '-999.000', ],
        'stdev_chl_aqua': [
            '-999.000', ],
        'chl_anomaly_aqua': [
            '-999.000', ],
        'pct_valid_chl_aqua': [
            '-999.000', ],
        'cdm_mean_seawifs': [
            '-999.000', ],
        'cdm_stdev_seawifs': [
            '-999.000', ],
        'cdm_anomaly_seawifs': [
            '-999.000', ],
        'pct_valid_cdm_seawifs': [
            '-999.000', ],
        'cdm_mean_aqua': [
            '-999.000', ],
        'cdm_stdev_aqua': [
            '-999.000', ],
        'cdm_anomaly_aqua': [
            '-999.000', ],
        'pct_valid_cdm_aqua': [
            '-999.000', ],
        'mean_bbp_aqua': [
            '-999.000', ],
        'stdev_bbp_aqua': [
            '-999.000', ],
        'bbp_anomaly_aqua': [
            '-999.000', ],
        'pct_valid_bbp_aqua': [
            '-999.000', ],
        'mean_nsst_aqua': [
            '-999.000', ],
        'stdev_nsst_aqua': [
            '-999.000', ],
        'nsst_anomaly_aqua': [
            '-999', ],
        'pct_valid_nsst_aqua': [
            '-999.000', ], }

                  )
# Coerce the data into the types specified in the metadata
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt2 = dt2.assign(Year_datetime=pd.to_datetime(dt2.Year, errors='coerce'))
dt2.Month = str(dt2.Month)
dt2.first_day = str(dt2.first_day)
dt2.last_day = str(dt2.last_day)
dt2.mean_chl_seawifs = pd.to_numeric(dt2.mean_chl_seawifs, errors='coerce')
dt2.stdev_chl_seawifs = pd.to_numeric(dt2.stdev_chl_seawifs, errors='coerce')
dt2.chl_anomaly_seawifs = pd.to_numeric(dt2.chl_anomaly_seawifs, errors='coerce')
dt2.pct_valid_chl_seawifs = pd.to_numeric(dt2.pct_valid_chl_seawifs, errors='coerce')
dt2.mean_chl_aqua = pd.to_numeric(dt2.mean_chl_aqua, errors='coerce')
dt2.stdev_chl_aqua = pd.to_numeric(dt2.stdev_chl_aqua, errors='coerce')
dt2.chl_anomaly_aqua = pd.to_numeric(dt2.chl_anomaly_aqua, errors='coerce')
dt2.pct_valid_chl_aqua = pd.to_numeric(dt2.pct_valid_chl_aqua, errors='coerce')
dt2.cdm_mean_seawifs = pd.to_numeric(dt2.cdm_mean_seawifs, errors='coerce')
dt2.cdm_stdev_seawifs = pd.to_numeric(dt2.cdm_stdev_seawifs, errors='coerce')
dt2.cdm_anomaly_seawifs = pd.to_numeric(dt2.cdm_anomaly_seawifs, errors='coerce')
dt2.pct_valid_cdm_seawifs = pd.to_numeric(dt2.pct_valid_cdm_seawifs, errors='coerce')
dt2.cdm_mean_aqua = pd.to_numeric(dt2.cdm_mean_aqua, errors='coerce')
dt2.cdm_stdev_aqua = pd.to_numeric(dt2.cdm_stdev_aqua, errors='coerce')
dt2.cdm_anomaly_aqua = pd.to_numeric(dt2.cdm_anomaly_aqua, errors='coerce')
dt2.pct_valid_cdm_aqua = pd.to_numeric(dt2.pct_valid_cdm_aqua, errors='coerce')
dt2.mean_bbp_aqua = pd.to_numeric(dt2.mean_bbp_aqua, errors='coerce')
dt2.stdev_bbp_aqua = pd.to_numeric(dt2.stdev_bbp_aqua, errors='coerce')
dt2.bbp_anomaly_aqua = pd.to_numeric(dt2.bbp_anomaly_aqua, errors='coerce')
dt2.pct_valid_bbp_aqua = pd.to_numeric(dt2.pct_valid_bbp_aqua, errors='coerce')
dt2.mean_nsst_aqua = pd.to_numeric(dt2.mean_nsst_aqua, errors='coerce')
dt2.stdev_nsst_aqua = pd.to_numeric(dt2.stdev_nsst_aqua, errors='coerce')
dt2.nsst_anomaly_aqua = pd.to_numeric(dt2.nsst_anomaly_aqua, errors='coerce')
dt2.pct_valid_nsst_aqua = pd.to_numeric(dt2.pct_valid_nsst_aqua, errors='coerce')

print("Here is a description of the data frame dt2 and number of lines\n")
print(dt2.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt2\n")
print(dt2.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt2.Year.describe())
print("--------------------\n\n")

print(dt2.Month.describe())
print("--------------------\n\n")

print(dt2.first_day.describe())
print("--------------------\n\n")

print(dt2.last_day.describe())
print("--------------------\n\n")

print(dt2.mean_chl_seawifs.describe())
print("--------------------\n\n")

print(dt2.stdev_chl_seawifs.describe())
print("--------------------\n\n")

print(dt2.chl_anomaly_seawifs.describe())
print("--------------------\n\n")

print(dt2.pct_valid_chl_seawifs.describe())
print("--------------------\n\n")

print(dt2.mean_chl_aqua.describe())
print("--------------------\n\n")

print(dt2.stdev_chl_aqua.describe())
print("--------------------\n\n")

print(dt2.chl_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt2.pct_valid_chl_aqua.describe())
print("--------------------\n\n")

print(dt2.cdm_mean_seawifs.describe())
print("--------------------\n\n")

print(dt2.cdm_stdev_seawifs.describe())
print("--------------------\n\n")

print(dt2.cdm_anomaly_seawifs.describe())
print("--------------------\n\n")

print(dt2.pct_valid_cdm_seawifs.describe())
print("--------------------\n\n")

print(dt2.cdm_mean_aqua.describe())
print("--------------------\n\n")

print(dt2.cdm_stdev_aqua.describe())
print("--------------------\n\n")

print(dt2.cdm_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt2.pct_valid_cdm_aqua.describe())
print("--------------------\n\n")

print(dt2.mean_bbp_aqua.describe())
print("--------------------\n\n")

print(dt2.stdev_bbp_aqua.describe())
print("--------------------\n\n")

print(dt2.bbp_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt2.pct_valid_bbp_aqua.describe())
print("--------------------\n\n")

print(dt2.mean_nsst_aqua.describe())
print("--------------------\n\n")

print(dt2.stdev_nsst_aqua.describe())
print("--------------------\n\n")

print(dt2.nsst_anomaly_aqua.describe())
print("--------------------\n\n")

print(dt2.pct_valid_nsst_aqua.describe())
print("--------------------\n\n")

infile3 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/5/25/d6a2d263b446b46bef1f2b4f04545c07".strip()
infile3 = infile3.replace("https://", "http://")

dt3 = pd.read_csv(infile3
                  , skiprows=1
                  , sep=","
                  , names=[
        "seq_day",
        "date",
        "sst_mean",
        "sst_anomaly"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'seq_day':'int' ,
                  #             'date':'str' ,
                  #             'sst_mean':'float' ,
                  #             'sst_anomaly':'float'
                  #        }
                  , parse_dates=[
        'date',
    ]
                  , na_values={
        'sst_mean': [
            '-999', ],
        'sst_anomaly': [
            '-999', ], }

                  )
# Coerce the data into the types specified in the metadata
dt3.seq_day = pd.to_numeric(dt3.seq_day, errors='coerce', downcast='integer')
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt3 = dt3.assign(date_datetime=pd.to_datetime(dt3.date, errors='coerce'))
dt3.sst_mean = pd.to_numeric(dt3.sst_mean, errors='coerce')
dt3.sst_anomaly = pd.to_numeric(dt3.sst_anomaly, errors='coerce')

print("Here is a description of the data frame dt3 and number of lines\n")
print(dt3.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt3\n")
print(dt3.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt3.seq_day.describe())
print("--------------------\n\n")

print(dt3.date.describe())
print("--------------------\n\n")

print(dt3.sst_mean.describe())
print("--------------------\n\n")

print(dt3.sst_anomaly.describe())
print("--------------------\n\n")
