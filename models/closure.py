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
closure_data = pd.read_csv("C:/Users/pirtapalola/Documents/iop_data/data/closure_experiment_600_705nm.csv")

# List containing the unique IDs of the sampling sites
sites = ['ONE07', 'ONE08', 'ONE09', 'ONE10', 'ONE11', 'ONE12',
         'RIM01', 'RIM02', 'RIM03', 'RIM04', 'RIM05', 'RIM06']
benthic_sites = [x+'_benthic' for x in sites]
surface_sites = [x+'_surface' for x in sites]
lee99_sites = [x+'_L99' for x in sites]
maritorena94_sites = [x+'_M94' for x in sites]


def make_dataframe(dataframe, column_name):
    new_dataframe = pd.DataFrame()
    for x in column_name:
        new_dataframe[x] = dataframe[x]
    return new_dataframe


benthic_data = make_dataframe(closure_data, benthic_sites)
surface_data = make_dataframe(closure_data, surface_sites)
lee_data = make_dataframe(closure_data, lee99_sites)
maritorena_data = make_dataframe(closure_data, maritorena94_sites)


# This function creates a dictionary
def make_dictionary(dataframe, column_name):
    data_dict = {}
    for y in column_name:
        data_dict[y] = np.array(dataframe[y])
    return data_dict


# Apply the function to create separate dictionaries for the different datasets
benthic_dictionary = make_dictionary(closure_data, benthic_sites)
surface_dictionary = make_dictionary(closure_data, surface_sites)
lee_dictionary = make_dictionary(closure_data, lee99_sites)
maritorena_dictionary = make_dictionary(closure_data, maritorena94_sites)


def compare_results(dict1, dict2, site_list, str1, str2):
    sam_result_list = []
    for n in site_list:
        sam_result_list.append(pysptools.distance.SAM(dict1[(n+str1)], dict2[n+str2]))
    return sam_result_list


# Create a dataframe to store the results
spectral_similarity = pd.DataFrame()
spectral_similarity["Unique_ID"] = sites

# Comparison of surface and benthic measurements
surface_benthic = compare_results(surface_dictionary, benthic_dictionary, sites, '_surface', '_benthic')
spectral_similarity["Surface_Benthic"] = surface_benthic  # Add the results in the dataframe as a column

# Comparison of surface measurements and the Maritorena et al. (1994) forward model results
surface_maritorena = compare_results(surface_dictionary, maritorena_dictionary, sites, '_surface', '_M94')
spectral_similarity["Surface_Maritorena"] = surface_maritorena  # Add the results in the dataframe as a column

# Comparison of surface measurements and the Lee et al. (1999) forward model results
surface_lee = compare_results(surface_dictionary, lee_dictionary, sites, '_surface', '_L99')
spectral_similarity["Surface_Lee"] = surface_lee  # Add the results in the dataframe as a column

print(spectral_similarity)
spectral_similarity.to_csv('C:/Users/pirtapalola/Documents/iop_data/spectral_similarity_600_705nm.csv')
