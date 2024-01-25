# Package ID: knb-lter-mcr.1034.9 Cataloging System:https://pasta.edirepository.org.
# Data set title: MCR LTER: Coral Reef: Water Column: Nutrients, ongoing since 2005.
# Data set creator:    - Moorea Coral Reef LTER
# Data set creator:  Alice Alldredge - Moorea Coral Reef LTER
# Contact:    - Information Manager Moorea Coral Reef LTER  - mcrlter@msi.ucsb.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu
#
# This program creates numbered PANDA dataframes named dt1,dt2,dt3...,
# one for each data table in the dataset. It also provides some basic
# summaries of their contents. NumPy and Pandas modules need to be installed
# for the program to run.

import numpy as np
import pandas as pd

infile1 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/1034/9/4dc6aeb4def7563e3c769f27473933e0".strip()
infile1 = infile1.replace("https://", "http://")

dt1 = pd.read_csv(infile1
                  , skiprows=1
                  , sep=","
                  , quotechar='"'
                  , names=[
        "Cruise",
        "Location",
        "Type",
        "Date",
        "Time",
        "Latitude",
        "Longitude",
        "Bottom_Depth",
        "Sample_Depth",
        "Phosphate",
        "Silicate",
        "Nitrite",
        "Nitrite_and_Nitrate",
        "Comments"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'Cruise':'str' ,
                  #             'Location':'str' ,
                  #             'Type':'str' ,
                  #             'Date':'str' ,
                  #             'Time':'str' ,
                  #             'Latitude':'float' ,
                  #             'Longitude':'float' ,
                  #             'Bottom_Depth':'float' ,
                  #             'Sample_Depth':'float' ,
                  #             'Phosphate':'float' ,
                  #             'Silicate':'float' ,
                  #             'Nitrite':'float' ,
                  #             'Nitrite_and_Nitrate':'float' ,
                  #             'Comments':'str'
                  #        }
                  , parse_dates=[
        'Date',
        'Time',
    ]
                  , na_values={
        'Time': [
            'NA', ],
        'Latitude': [
            'NA', ],
        'Longitude': [
            'NA', ],
        'Bottom_Depth': [
            'NA', ],
        'Sample_Depth': [
            'NA', ],
        'Phosphate': [
            'NA', ],
        'Silicate': [
            'NA', ],
        'Nitrite': [
            'NA', ],
        'Nitrite_and_Nitrate': [
            'NA', ],
        'Comments': [
            'na', ], }

                  )
# Coerce the data into the types specified in the metadata
dt1.Cruise = dt1.Cruise.astype('category')
dt1.Location = dt1.Location.astype('category')
dt1.Type = dt1.Type.astype('category')
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt1 = dt1.assign(Date_datetime=pd.to_datetime(dt1.Date, errors='coerce'))
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt1 = dt1.assign(Time_datetime=pd.to_datetime(dt1.Time, errors='coerce'))
dt1.Latitude = pd.to_numeric(dt1.Latitude, errors='coerce')
dt1.Longitude = pd.to_numeric(dt1.Longitude, errors='coerce')
dt1.Bottom_Depth = pd.to_numeric(dt1.Bottom_Depth, errors='coerce')
dt1.Sample_Depth = pd.to_numeric(dt1.Sample_Depth, errors='coerce')
dt1.Phosphate = pd.to_numeric(dt1.Phosphate, errors='coerce')
dt1.Silicate = pd.to_numeric(dt1.Silicate, errors='coerce')
dt1.Nitrite = pd.to_numeric(dt1.Nitrite, errors='coerce')
dt1.Nitrite_and_Nitrate = pd.to_numeric(dt1.Nitrite_and_Nitrate, errors='coerce')
dt1.Comments = dt1.Comments.astype('category')

print("Here is a description of the data frame dt1 and number of lines\n")
print(dt1.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt1\n")
print(dt1.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt1.Cruise.describe())
print("--------------------\n\n")

print(dt1.Location.describe())
print("--------------------\n\n")

print(dt1.Type.describe())
print("--------------------\n\n")

print(dt1.Date.describe())
print("--------------------\n\n")

print(dt1.Time.describe())
print("--------------------\n\n")

print(dt1.Latitude.describe())
print("--------------------\n\n")

print(dt1.Longitude.describe())
print("--------------------\n\n")

print(dt1.Bottom_Depth.describe())
print("--------------------\n\n")

print(dt1.Sample_Depth.describe())
print("--------------------\n\n")

print(dt1.Phosphate.describe())
print("--------------------\n\n")

print(dt1.Silicate.describe())
print("--------------------\n\n")

print(dt1.Nitrite.describe())
print("--------------------\n\n")

print(dt1.Nitrite_and_Nitrate.describe())
print("--------------------\n\n")

print(dt1.Comments.describe())
print("--------------------\n\n")
