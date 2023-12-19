"""

Create the runlist.txt file for the Hydrolight simulations.

STEP 1. Save the input file names into a list.
STEP 2. Write the runlist file containing all the file names.

Last updated on 19 December 2023 by Pirta Palola

"""

# Import libraries

import os

""""STEP 1. Save the input file names into a list."""

os.chdir(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup')
the_list = []

for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup'):
    for file in files:
        if file.endswith('.txt'):
            the_list.append(file)


"""STEP 2. Write the runlist file containing all the file names."""


def new_runlist_file(path_list):
    # open file in write mode
    path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/runlist.txt'
    with open(path, 'w') as fp:
        for item in path_list:
            fp.write(item + ' \n')
    return path_list


new_runlist_file(the_list)
