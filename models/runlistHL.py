"""This code creates the runlist.txt file for the Hydrolight simulations."""

import os

# Access the input files
os.chdir(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_sbi_setup')
the_list = []

for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_sbi_setup'):
    for file in files:
        if file.endswith('.txt'):
            the_list.append(file)


def new_runlist_file(path_list):
    # open file in write mode
    path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_final_setup/runlist.txt'
    with open(path, 'w') as fp:
        for item in path_list:
            fp.write(item + ' \n')
    return path_list


new_runlist_file(the_list)
