# Package ID: knb-lter-mcr.10.36 Cataloging System:https://pasta.edirepository.org.
# Data set title: MCR LTER: Coral Reef: Water Column: Nearshore Water Profiles, CTD, Primary Production, and Chemistry
# ongoing since 2005.
# Data set creator:    - Moorea Coral Reef LTER
# Data set creator:  Alice Alldredge - Moorea Coral Reef LTER 
# Data set creator:  Craig Carlson - Moorea Coral Reef LTER
# Contact:    - Information Manager Moorea Coral Reef LTER  - mcrlter@msi.ucsb.edu
# Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu
#
# This program creates numbered PANDA dataframes named dt1,dt2,dt3...,
# one for each data table in the dataset. It also provides some basic
# summaries of their contents. NumPy and Pandas modules need to be installed
# for the program to run.

import numpy as np
import pandas as pd

infile1 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/10/36/d3c2de156d7afbb332786e78b8c0877f".strip()
infile1 = infile1.replace("https://", "http://")

dt1 = pd.read_csv(infile1
                  , skiprows=1
                  , sep=";"
                  , names=[
        "Cruise",
        "Station_code",
        "Type",
        "Date",
        "Time",
        "Longitude",
        "Latitude",
        "Bottom_Depth",
        "Depth",
        "Decimal_Year",
        "Pressure_avg",
        "Pressure_Bin",
        "Temperature",
        "Conductivity",
        "V0",
        "V1",
        "V2",
        "Descent_Rate",
        "Oxygen",
        "Fluor",
        "Turbidity",
        "Potential_Temperature",
        "Salinity",
        "Density_Anomaly",
        "Specific_Volume_Anomaly",
        "Scans_Per_Bin",
        "Location"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'Cruise':'str' ,
                  #             'Station_code':'str' ,
                  #             'Type':'str' ,
                  #             'Date':'str' ,
                  #             'Time':'str' ,
                  #             'Longitude':'float' ,
                  #             'Latitude':'float' ,
                  #             'Bottom_Depth':'float' ,
                  #             'Depth':'float' ,
                  #             'Decimal_Year':'float' ,
                  #             'Pressure_avg':'float' ,
                  #             'Pressure_Bin':'float' ,
                  #             'Temperature':'float' ,
                  #             'Conductivity':'float' ,
                  #             'V0':'float' ,
                  #             'V1':'float' ,
                  #             'V2':'float' ,
                  #             'Descent_Rate':'float' ,
                  #             'Oxygen':'float' ,
                  #             'Fluor':'float' ,
                  #             'Turbidity':'float' ,
                  #             'Potential_Temperature':'float' ,
                  #             'Salinity':'float' ,
                  #             'Density_Anomaly':'float' ,
                  #             'Specific_Volume_Anomaly':'float' ,
                  #             'Scans_Per_Bin':'int' ,
                  #             'Location':'str'
                  #        }
                  , parse_dates=[
        'Date',
        'Time',
    ]
                  , na_values={
        'Cruise': [
            '99999', ],
        'Station_code': [
            '99999', ],
        'Type': [
            '99999', ],
        'Date': [
            '99999', ],
        'Time': [
            '99999', ],
        'Longitude': [
            '99999',
            'NaN', ],
        'Latitude': [
            '99999',
            'NaN', ],
        'Bottom_Depth': [
            '99999', ],
        'Depth': [
            '99999', ],
        'Decimal_Year': [
            '99999', ],
        'Pressure_avg': [
            '99999', ],
        'Pressure_Bin': [
            '99999', ],
        'Temperature': [
            '99999', ],
        'Conductivity': [
            '99999', ],
        'V0': [
            '99999', ],
        'V1': [
            '99999', ],
        'V2': [
            '99999', ],
        'Descent_Rate': [
            '99999', ],
        'Oxygen': [
            '99999', ],
        'Fluor': [
            '99999', ],
        'Turbidity': [
            '99999', ],
        'Potential_Temperature': [
            '99999', ],
        'Salinity': [
            '99999', ],
        'Density_Anomaly': [
            '99999', ],
        'Specific_Volume_Anomaly': [
            '99999', ],
        'Scans_Per_Bin': [
            '99999', ],
        'Location': [
            '99999', ], }

                  )
# Coerce the data into the types specified in the metadata
dt1.Cruise = dt1.Cruise.astype('category')
dt1.Station_code = dt1.Station_code.astype('category')
dt1.Type = dt1.Type.astype('category')
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt1 = dt1.assign(Date_datetime=pd.to_datetime(dt1.Date, errors='coerce'))
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt1 = dt1.assign(Time_datetime=pd.to_datetime(dt1.Time, errors='coerce'))
dt1.Longitude = pd.to_numeric(dt1.Longitude, errors='coerce')
dt1.Latitude = pd.to_numeric(dt1.Latitude, errors='coerce')
dt1.Bottom_Depth = pd.to_numeric(dt1.Bottom_Depth, errors='coerce')
dt1.Depth = pd.to_numeric(dt1.Depth, errors='coerce')
dt1.Decimal_Year = pd.to_numeric(dt1.Decimal_Year, errors='coerce')
dt1.Pressure_avg = pd.to_numeric(dt1.Pressure_avg, errors='coerce')
dt1.Pressure_Bin = pd.to_numeric(dt1.Pressure_Bin, errors='coerce')
dt1.Temperature = pd.to_numeric(dt1.Temperature, errors='coerce')
dt1.Conductivity = pd.to_numeric(dt1.Conductivity, errors='coerce')
dt1.V0 = pd.to_numeric(dt1.V0, errors='coerce')
dt1.V1 = pd.to_numeric(dt1.V1, errors='coerce')
dt1.V2 = pd.to_numeric(dt1.V2, errors='coerce')
dt1.Descent_Rate = pd.to_numeric(dt1.Descent_Rate, errors='coerce')
dt1.Oxygen = pd.to_numeric(dt1.Oxygen, errors='coerce')
dt1.Fluor = pd.to_numeric(dt1.Fluor, errors='coerce')
dt1.Turbidity = pd.to_numeric(dt1.Turbidity, errors='coerce')
dt1.Potential_Temperature = pd.to_numeric(dt1.Potential_Temperature, errors='coerce')
dt1.Salinity = pd.to_numeric(dt1.Salinity, errors='coerce')
dt1.Density_Anomaly = pd.to_numeric(dt1.Density_Anomaly, errors='coerce')
dt1.Specific_Volume_Anomaly = pd.to_numeric(dt1.Specific_Volume_Anomaly, errors='coerce')
dt1.Scans_Per_Bin = pd.to_numeric(dt1.Scans_Per_Bin, errors='coerce', downcast='integer')
dt1.Location = dt1.Location.astype('category')

print("Here is a description of the data frame dt1 and number of lines\n")
print(dt1.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt1\n")
print(dt1.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt1.Cruise.describe())
print("--------------------\n\n")

print(dt1.Station_code.describe())
print("--------------------\n\n")

print(dt1.Type.describe())
print("--------------------\n\n")

print(dt1.Date.describe())
print("--------------------\n\n")

print(dt1.Time.describe())
print("--------------------\n\n")

print(dt1.Longitude.describe())
print("--------------------\n\n")

print(dt1.Latitude.describe())
print("--------------------\n\n")

print(dt1.Bottom_Depth.describe())
print("--------------------\n\n")

print(dt1.Depth.describe())
print("--------------------\n\n")

print(dt1.Decimal_Year.describe())
print("--------------------\n\n")

print(dt1.Pressure_avg.describe())
print("--------------------\n\n")

print(dt1.Pressure_Bin.describe())
print("--------------------\n\n")

print(dt1.Temperature.describe())
print("--------------------\n\n")

print(dt1.Conductivity.describe())
print("--------------------\n\n")

print(dt1.V0.describe())
print("--------------------\n\n")

print(dt1.V1.describe())
print("--------------------\n\n")

print(dt1.V2.describe())
print("--------------------\n\n")

print(dt1.Descent_Rate.describe())
print("--------------------\n\n")

print(dt1.Oxygen.describe())
print("--------------------\n\n")

print(dt1.Fluor.describe())
print("--------------------\n\n")

print(dt1.Turbidity.describe())
print("--------------------\n\n")

print(dt1.Potential_Temperature.describe())
print("--------------------\n\n")

print(dt1.Salinity.describe())
print("--------------------\n\n")

print(dt1.Density_Anomaly.describe())
print("--------------------\n\n")

print(dt1.Specific_Volume_Anomaly.describe())
print("--------------------\n\n")

print(dt1.Scans_Per_Bin.describe())
print("--------------------\n\n")

print(dt1.Location.describe())
print("--------------------\n\n")

infile2 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/10/36/63f84a040d131723c60b14106d158578".strip()
infile2 = infile2.replace("https://", "http://")

dt2 = pd.read_csv(infile2
                  , skiprows=1
                  , sep=","
                  , quotechar='"'
                  , names=[
        "Cruise",
        "Cruise_code",
        "Location",
        "Type",
        "Date",
        "Time",
        "Latitude",
        "Longitude",
        "Bottom_Depth",
        "Sample_Depth",
        "Sample_Number",
        "Pressure_bins",
        "Temperature",
        "Salinity",
        "Sigma_theta",
        "Fluorescence",
        "Turbidity",
        "Phosphate",
        "Silicate",
        "Nitrite",
        "Nitrite_and_Nitrate",
        "POC",
        "PON",
        "TOC",
        "TOC_SD",
        "Alkalinity",
        "DIC",
        "Chlorophyll_a",
        "Phaeopigments",
        "Primary_Production",
        "Bacteria_Abundance",
        "Stained_Cell_Abundance",
        "Autofluorescent_Cell_Abundance",
        "Comments"]
                  # data type checking is commented out because it may cause data
                  # loads to fail if the data contains inconsistent values. Uncomment
                  # the following lines to enable data type checking

                  #            ,dtype={
                  #             'Cruise':'str' ,
                  #             'Cruise_code':'str' ,
                  #             'Location':'str' ,
                  #             'Type':'str' ,
                  #             'Date':'str' ,
                  #             'Time':'str' ,
                  #             'Latitude':'float' ,
                  #             'Longitude':'float' ,
                  #             'Bottom_Depth':'float' ,
                  #             'Sample_Depth':'float' ,
                  #             'Sample_Number':'str',
                  #             'Pressure_bins':'float' ,
                  #             'Temperature':'float' ,
                  #             'Salinity':'float' ,
                  #             'Sigma_theta':'float' ,
                  #             'Fluorescence':'float' ,
                  #             'Turbidity':'float' ,
                  #             'Phosphate':'float' ,
                  #             'Silicate':'float' ,
                  #             'Nitrite':'float' ,
                  #             'Nitrite_and_Nitrate':'float' ,
                  #             'POC':'float' ,
                  #             'PON':'float' ,
                  #             'TOC':'float' ,
                  #             'TOC_SD':'float' ,
                  #             'Alkalinity':'float' ,
                  #             'DIC':'float' ,
                  #             'Chlorophyll_a':'float' ,
                  #             'Phaeopigments':'float' ,
                  #             'Primary_Production':'float' ,
                  #             'Bacteria_Abundance':'float' ,
                  #             'Stained_Cell_Abundance':'float' ,
                  #             'Autofluorescent_Cell_Abundance':'float' ,
                  #             'Comments':'str'
                  #        }
                  , parse_dates=[
        'Date',
    ]
                  , na_values={
        'Time': [
            'NA', ],
        'Bottom_Depth': [
            'NA', ],
        'Pressure_bins': [
            'NA', ],
        'Temperature': [
            'NA',
            'pending', ],
        'Salinity': [
            'NA',
            'pending', ],
        'Sigma_theta': [
            'NA',
            'pending', ],
        'Fluorescence': [
            'NA',
            'pending', ],
        'Turbidity': [
            'NA',
            'pending', ],
        'Phosphate': [
            'NA', ],
        'Silicate': [
            'NA', ],
        'Nitrite': [
            'NA', ],
        'Nitrite_and_Nitrate': [
            'NA', ],
        'POC': [
            'NA', ],
        'PON': [
            'NA', ],
        'TOC': [
            'NA',
            'NA', ],
        'TOC_SD': [
            'NA',
            'NA', ],
        'Alkalinity': [
            'NA',
            'NA', ],
        'DIC': [
            'NA',
            'NA', ],
        'Chlorophyll_a': [
            'NA', ],
        'Phaeopigments': [
            'NA', ],
        'Primary_Production': [
            'NA', ],
        'Bacteria_Abundance': [
            'NA', ],
        'Stained_Cell_Abundance': [
            'NA', ],
        'Autofluorescent_Cell_Abundance': [
            'NA',
            'NA', ],
        'Comments': [
            'NA', ], }

                  )
# Coerce the data into the types specified in the metadata
dt2.Cruise = dt2.Cruise.astype('category')
dt2.Cruise_code = dt2.Cruise_code.astype('category')
dt2.Location = dt2.Location.astype('category')
dt2.Type = dt2.Type.astype('category')
# Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
# This new column is added to the dataframe but does not show up in automated summaries below.
dt2 = dt2.assign(Date_datetime=pd.to_datetime(dt2.Date, errors='coerce'))
dt2.Time = dt2.Time.astype('category')
dt2.Latitude = pd.to_numeric(dt2.Latitude, errors='coerce')
dt2.Longitude = pd.to_numeric(dt2.Longitude, errors='coerce')
dt2.Bottom_Depth = pd.to_numeric(dt2.Bottom_Depth, errors='coerce')
dt2.Sample_Depth = pd.to_numeric(dt2.Sample_Depth, errors='coerce')
dt2.Sample_Number = str(dt2.Sample_Number)
dt2.Pressure_bins = pd.to_numeric(dt2.Pressure_bins, errors='coerce')
dt2.Temperature = pd.to_numeric(dt2.Temperature, errors='coerce')
dt2.Salinity = pd.to_numeric(dt2.Salinity, errors='coerce')
dt2.Sigma_theta = pd.to_numeric(dt2.Sigma_theta, errors='coerce')
dt2.Fluorescence = pd.to_numeric(dt2.Fluorescence, errors='coerce')
dt2.Turbidity = pd.to_numeric(dt2.Turbidity, errors='coerce')
dt2.Phosphate = pd.to_numeric(dt2.Phosphate, errors='coerce')
dt2.Silicate = pd.to_numeric(dt2.Silicate, errors='coerce')
dt2.Nitrite = pd.to_numeric(dt2.Nitrite, errors='coerce')
dt2.Nitrite_and_Nitrate = pd.to_numeric(dt2.Nitrite_and_Nitrate, errors='coerce')
dt2.POC = pd.to_numeric(dt2.POC, errors='coerce')
dt2.PON = pd.to_numeric(dt2.PON, errors='coerce')
dt2.TOC = pd.to_numeric(dt2.TOC, errors='coerce')
dt2.TOC_SD = pd.to_numeric(dt2.TOC_SD, errors='coerce')
dt2.Alkalinity = pd.to_numeric(dt2.Alkalinity, errors='coerce')
dt2.DIC = pd.to_numeric(dt2.DIC, errors='coerce')
dt2.Chlorophyll_a = pd.to_numeric(dt2.Chlorophyll_a, errors='coerce')
dt2.Phaeopigments = pd.to_numeric(dt2.Phaeopigments, errors='coerce')
dt2.Primary_Production = pd.to_numeric(dt2.Primary_Production, errors='coerce')
dt2.Bacteria_Abundance = pd.to_numeric(dt2.Bacteria_Abundance, errors='coerce')
dt2.Stained_Cell_Abundance = pd.to_numeric(dt2.Stained_Cell_Abundance, errors='coerce')
dt2.Autofluorescent_Cell_Abundance = pd.to_numeric(dt2.Autofluorescent_Cell_Abundance, errors='coerce')
dt2.Comments = dt2.Comments.astype('category')

print("Here is a description of the data frame dt2 and number of lines\n")
print(dt2.info())
print("--------------------\n\n")
print("Here is a summary of numerical variables in the data frame dt2\n")
print(dt2.describe())
print("--------------------\n\n")

print("The analyses below are basic descriptions of the variables. After testing, they should be replaced.\n")

print(dt2.Cruise.describe())
print("--------------------\n\n")

print(dt2.Cruise_code.describe())
print("--------------------\n\n")

print(dt2.Location.describe())
print("--------------------\n\n")

print(dt2.Type.describe())
print("--------------------\n\n")

print(dt2.Date.describe())
print("--------------------\n\n")

print(dt2.Time.describe())
print("--------------------\n\n")

print(dt2.Latitude.describe())
print("--------------------\n\n")

print(dt2.Longitude.describe())
print("--------------------\n\n")

print(dt2.Bottom_Depth.describe())
print("--------------------\n\n")

print(dt2.Sample_Depth.describe())
print("--------------------\n\n")

print(dt2.Sample_Number.describe())
print("--------------------\n\n")

print(dt2.Pressure_bins.describe())
print("--------------------\n\n")

print(dt2.Temperature.describe())
print("--------------------\n\n")

print(dt2.Salinity.describe())
print("--------------------\n\n")

print(dt2.Sigma_theta.describe())
print("--------------------\n\n")

print(dt2.Fluorescence.describe())
print("--------------------\n\n")

print(dt2.Turbidity.describe())
print("--------------------\n\n")

print(dt2.Phosphate.describe())
print("--------------------\n\n")

print(dt2.Silicate.describe())
print("--------------------\n\n")

print(dt2.Nitrite.describe())
print("--------------------\n\n")

print(dt2.Nitrite_and_Nitrate.describe())
print("--------------------\n\n")

print(dt2.POC.describe())
print("--------------------\n\n")

print(dt2.PON.describe())
print("--------------------\n\n")

print(dt2.TOC.describe())
print("--------------------\n\n")

print(dt2.TOC_SD.describe())
print("--------------------\n\n")

print(dt2.Alkalinity.describe())
print("--------------------\n\n")

print(dt2.DIC.describe())
print("--------------------\n\n")

print(dt2.Chlorophyll_a.describe())
print("--------------------\n\n")

print(dt2.Phaeopigments.describe())
print("--------------------\n\n")

print(dt2.Primary_Production.describe())
print("--------------------\n\n")

print(dt2.Bacteria_Abundance.describe())
print("--------------------\n\n")

print(dt2.Stained_Cell_Abundance.describe())
print("--------------------\n\n")

print(dt2.Autofluorescent_Cell_Abundance.describe())
print("--------------------\n\n")

print(dt2.Comments.describe())
print("--------------------\n\n")
