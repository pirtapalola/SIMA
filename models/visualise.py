import os
import pandas as pd

os.chdir(r'C:/Users/pirtapalola/Documents/iop_data/data/L99_data_v2')
the_list = []
trios_list = []
kd_list = []
plymouth_absorption_list = []
plymouth_backscatter_list = []
m94_list = []
l99_list = []

# Create a list that contains the paths of all the csv files in a folder
"""for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/forward_model'):
    for file in files:
        if file.endswith('.csv'):
            the_list.append(file)

# Do the same for the folder with the trios data
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/trios_surface'):
    for file in files:
        if file.endswith('.csv'):
            trios_list.append(file)

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

# Do the same for the folder with the M94/L99 modelling results
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/iop_data/data/L99_data_v2'):
    for file in files:
        if file.endswith('.csv'):
            l99_list.append(file)


# Define a function that takes a list of paths, reads all the csv files,
# and adds the first column to a new pandas dataframe


def readdataframe(list_paths):
    df = pd.DataFrame()  # define df as an empty pandas DataFrame
    for element in list_paths:
        # print(element)
        df[element] = pd.read_csv(element, sep=',', header=0, usecols=[1])
    return df


fm_reflectance = readdataframe(the_list)
# fm_reflectance.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/fm_reflectance.csv')
surface_reflectance = readdataframe(trios_list)
#surface_reflectance.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/surface_reflectance.csv')
kd_dataframe = readdataframe(kd_list)
# kd_dataframe.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/kd_dataframe.csv')
total_a = readdataframe(plymouth_absorption_list)
# total_a.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/Plymouth_iop/total_a.csv')
total_b = readdataframe(plymouth_backscatter_list)
# total_b.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/Plymouth_iop/total_b.csv')
m94_results_df = readdataframe(m94_list)
# m94_results_df.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/output/M94_output.csv')
l99_results_df = readdataframe(l99_list)
l99_results_df.to_csv('C:/Users/pirtapalola/Documents/iop_data/data/output/L99_output_v2.csv')
