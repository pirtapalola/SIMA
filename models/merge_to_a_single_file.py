"""

Merge data from multiple files into a single csv file.

Last updated on 21 January 2024 by Pirta Palola

"""

# Import libraries
import os
import pandas as pd

# Define the folder
os.chdir(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
         r'In_water_calibration_2022/interpolated_files_2022')

# Create an empty list
the_list = []

# Create a list that contains the paths of all the csv files in a folder
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                                 r'In_water_calibration_2022/interpolated_files_2022'):
    for file in files:
        if file.endswith('.csv'):
            the_list.append(file)

# Define a function that takes a list of paths, reads all the csv files,
# and adds the first column to a new pandas dataframe


def readdataframe(list_paths):
    df = pd.DataFrame()  # define df as an empty pandas DataFrame
    for element in list_paths:
        # print(element)
        df[element] = pd.read_csv(element, sep=',', header=0, usecols=[2])
    return df


the_dataframe = readdataframe(the_list)

# Save the dataframe as a csv file
the_dataframe.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                     'In_water_calibration_2022/surface_reflectance_2022.csv')
