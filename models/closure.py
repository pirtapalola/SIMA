"""This code implements Spectral Angle Mapping (SAM) and Spectral Information Divergence (SID)
    to quantify the similarity of the measured and modelled spectra.

References (SAM):
    Dennison et al. (2004)
    Kruse et al. (1993)
    Schwarz & Staenz (2001)

SAM has been applied by, for example:
    Brando et al. (2009)
    Kutser et al. (2006)
    Petit et al. (2017)

References (SID):
    Chang (2000)
"""


# Import libraries
import pandas as pd
import numpy as np
import pysptools.distance

# Read the csv data into a pandas dataframe
closure_data = pd.read_csv("C:/Users/pirtapalola/Documents/iop_data/data/closure_experiment.csv")

# List containing the unique IDs of the sampling sites
sites = ['ONE07', 'ONE08', 'ONE09', 'ONE10', 'ONE11', 'ONE12',
         'RIM01', 'RIM02', 'RIM03', 'RIM04', 'RIM05', 'RIM06']
benthic_sites = [x+'_benthic' for x in sites]
surface_sites = [x+'_surface' for x in sites]
lee99_sites = [x+'_L99' for x in sites]
maritorena94_sites = [x+'_M94' for x in sites]


# This function creates a dictionary with Unique ID as key
def make_dictionary(dataframe, site_list, column_name):
    data_dict = {}
    for i in site_list:
        for n in column_name:
            data_dict[i] = np.array(dataframe[n])
    return data_dict


# Apply the function to create separate dictionaries for the different datasets
benthic_dictionary = make_dictionary(closure_data, sites, benthic_sites)
surface_dictionary = make_dictionary(closure_data, sites, surface_sites)
lee_dictionary = make_dictionary(closure_data, sites, lee99_sites)
maritorena_dictionary = make_dictionary(closure_data, sites, maritorena94_sites)
print(len(benthic_dictionary['ONE07']))

# Create a dataframe to store the results
spectral_similarity = pd.DataFrame()
spectral_similarity["Unique_ID"] = sites

# Add the results in the dataframe as a column
# Comparison of surface measurements and the Lee et al. (1999) forward model results
surface_lee_comparison = []
for i in sites:
    sam_result = pysptools.distance.SAM(surface_dictionary[i], lee_dictionary[i])
    surface_lee_comparison.append(sam_result)
spectral_similarity["Surface_Lee"] = surface_lee_comparison

# Comparison of surface measurements and the Maritorena et al. (1994) forward model results
surface_maritorena_comparison = []
for i in sites:
    sam_result = pysptools.distance.SAM(surface_dictionary[i], maritorena_dictionary[i])
    surface_maritorena_comparison.append(sam_result)
spectral_similarity["Surface_Maritorena"] = surface_maritorena_comparison

# Comparison of surface measurements and benthic measurements
surface_benthic_comparison = []
for i in sites:
    sam_result = pysptools.distance.SAM(surface_dictionary[i], benthic_dictionary[i])
    surface_benthic_comparison.append(sam_result)
spectral_similarity["Surface_Benthic"] = surface_benthic_comparison

print(spectral_similarity)
spectral_similarity.to_csv('C:/Users/pirtapalola/Documents/iop_data/spectral_similarity.csv')
