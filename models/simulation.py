"""This code creates input files for HydroLight simulations"""

import pandas as pd
import itertools

# Read the file
# Specify which row(s) to import
specific_rows = [6]  # This row specifies the water constituent concentrations
# 1st element: water
# 2nd element: phytoplankton
# 3rd element: CDOM
# 4th element: SPM

concentrations = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/Icorals.txt',
                             sep=',', header=None, skiprows=lambda x: x not in specific_rows)
print(concentrations)

# Create lists that contain the different variations.
set1 = [1, 2, 3]

# Define a function that creates all the possible permutations of water constituent concentrations.


def permutations(variations, number_of_var):
    # Create all the possible permutations
    list1 = [permutation for permutation in itertools.permutations(variations, number_of_var)]
    # Remove duplicate permutations
    list2 = list(set(list1))
    return list2


data = permutations(set1, 3)
df = pd.DataFrame(data, columns=['phytoplankton', 'CDOM', 'SPM'])
print(df)
