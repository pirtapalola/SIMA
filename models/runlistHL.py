"""This code creates the runlist.txt file for the Hydrolight simulations."""

import os

# Access the input files
os.chdir(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_setup_files')
the_list = []

for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_setup_files'):
    for file in files:
        if file.endswith('.txt'):
            the_list.append(file)



