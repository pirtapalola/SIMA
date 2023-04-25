import os
import pandas as pd

os.chdir(r'C:/Users/pirtapalola/Documents/iop_data/data/trios')
the_list = []
trios_list = []
kd_list = []
plymouth_absorption_list = []
plymouth_backscatter_list = []

# Create a list that contains the paths of all the csv files in a folder
"""for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/forward_model'):
    for file in files:
        if file.endswith('.csv'):
            the_list.append(file)"""

# Do the same for the folder with the trios data
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/trios'):
    for file in files:
        if file.endswith('.csv'):
            trios_list.append(file)
"""
# Do the same for the folder with the diffuse attenuation coefficient data
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/kd_data'):
    for file in files:
        if file.endswith('_kd.csv'):
            kd_list.append(file)

# Do the same for the folder with the Plymouth total absorption data
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/Plymouth_iop/plymouth_water_iop'):
    for file in files:
        if file.endswith('a.csv'):
            plymouth_absorption_list.append(file)

# Do the same for the folder with the Plymouth total backscatter data
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/Plymouth_iop/plymouth_water_iop'):
    for file in files:
        if file.endswith('b.csv'):
            plymouth_backscatter_list.append(file)"""

# Define a function that takes a list of paths, reads all the csv files,
# and adds the first column to a new pandas dataframe


def readdataframe(list_paths):
    df = pd.DataFrame()  # define df as an empty pandas DataFrame
    for element in list_paths:
        # print(element)
        df[element] = pd.read_csv(element, sep=',', header=0, usecols=[2])
    return df


fm_reflectance = readdataframe(the_list)
# fm_reflectance.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/fm_reflectance.csv')
benthic_reflectance = readdataframe(trios_list)
benthic_reflectance.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/benthic_reflectance_all_sites.csv')
kd_dataframe = readdataframe(kd_list)
# kd_dataframe.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/kd_dataframe.csv')
total_a = readdataframe(plymouth_absorption_list)
# total_a.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/Plymouth_iop/total_a.csv')
total_b = readdataframe(plymouth_backscatter_list)
# total_b.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/Plymouth_iop/total_b.csv')
