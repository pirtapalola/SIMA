# Package ID: knb-lter-mcr.4011.1 Cataloging System:https://pasta.edirepository.org.
# Data set title: MCR LTER: Coral Reef: In Situ Light Data, ongoing since 2021.
# Data set creator:    - Moorea Coral Reef LTER
# Data set creator:  Peter Edmunds - Moorea Coral Reef LTER
# Metadata Provider:    - Moorea Coral Reef LTER
# Contact:    - Information Manager Moorea Coral Reef LTER  - mcrlter@msi.ucsb.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu
#
# This program creates numbered PANDA dataframes named dt1,dt2,dt3...,
# one for each data table in the dataset. It also provides some basic
# summaries of their contents. NumPy and Pandas modules need to be installed
# for the program to run.

import numpy as np
import pandas as pd

infile1 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/4011/1/c4dcd5f3657f38ddb856d51d4be78554".strip()
infile1 = infile1.replace("https://", "http://")

dt1 = pd.read_csv(infile1
                  , skiprows=1
                  , sep=","
                  , quotechar='"'
                  , names=[
        "date_time_local",
        "surface_max_par",
        "surface_Integrated",
        "depth_max_par",
        "depth_integrated",
        "kd_par"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'date_time_local':'str' ,
                  #             'surface_max_par':'float' ,
                  #             'surface_Integrated':'float' ,
                  #             'depth_max_par':'float' ,
                  #             'depth_integrated':'float' ,
                  #             'kd_par':'float'
                  #        }
                  , parse_dates=[
        'date_time_local',
    ]
                  , na_values={
        'surface_max_par': [
            'NA', ],
        'surface_Integrated': [
            'NA', ],
        'depth_max_par': [
            'NA', ],
        'depth_integrated': [
            'NA', ],
        'kd_par': [
            'NA', ], }

                  )
# Coerce the data into the types specified in the metadata
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt1 = dt1.assign(date_time_local_datetime=pd.to_datetime(dt1.date_time_local, errors='coerce'))
dt1.surface_max_par = pd.to_numeric(dt1.surface_max_par, errors='coerce')
dt1.surface_Integrated = pd.to_numeric(dt1.surface_Integrated, errors='coerce')
dt1.depth_max_par = pd.to_numeric(dt1.depth_max_par, errors='coerce')
dt1.depth_integrated = pd.to_numeric(dt1.depth_integrated, errors='coerce')
dt1.kd_par = pd.to_numeric(dt1.kd_par, errors='coerce')

print("Here is a description of the data frame dt1 and number of lines\n")
print(dt1.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt1\n")
print(dt1.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt1.date_time_local.describe())
print("--------------------\n\n")

print(dt1.surface_max_par.describe())
print("--------------------\n\n")

print(dt1.surface_Integrated.describe())
print("--------------------\n\n")

print(dt1.depth_max_par.describe())
print("--------------------\n\n")

print(dt1.depth_integrated.describe())
print("--------------------\n\n")

print(dt1.kd_par.describe())
print("--------------------\n\n")
