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


def make_numpy_arrays(dataframe, column_name):
    return np.array(dataframe[column_name])


test = make_numpy_arrays(closure_data, benthic_sites[0])
print(test)
s0 = np.array(closure_data['ONE07_benthic'])
s1 = np.array(closure_data['ONE07_surface'])
s2 = np.array(closure_data['ONE07_L99'])
s3 = np.array(closure_data['ONE07_M94'])

ONE07_L99_SAM = pysptools.distance.SAM(s1, s0)
ONE07_M94_SAM = pysptools.distance.SAM(s1, s3)
print(ONE07_L99_SAM)
print(ONE07_M94_SAM)
