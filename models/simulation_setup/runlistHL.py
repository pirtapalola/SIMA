"""

SIMULATION SET-UP I: Creating set-up files
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Create the runlist.txt file for the EcoLight simulations.
STEP 1. Save the input file names into a list.
STEP 2. Write the runlist file containing all the file names.

Last updated on 27 August 2024

"""

# Import libraries

import os

""""STEP 1. Save the input file names into a list."""

os.chdir(r'data/setup_files/')
the_list = []

for root, dirs, files in os.walk(r'data/setup_files/'):
    for file in files:
        if file.endswith('.txt'):
            the_list.append(file)


"""STEP 2. Write the runlist file containing all the file names."""


def new_runlist_file(path_list):
    # open file in write mode
    path = 'data/simulation_setup/runlist.txt'
    with open(path, 'w') as fp:
        for item in path_list:
            fp.write(item + ' \n')
    return path_list


new_runlist_file(the_list)
