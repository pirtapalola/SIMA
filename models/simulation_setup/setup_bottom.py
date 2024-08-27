"""

SIMULATION SET-UP II: Changing the bottom reflectance file name in the EcoLight setup files.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

STEP 1. Create a list of all the EcoLight files to be modified.
STEP 2. Write the new EcoLight set-up files.
STEP 3. Check that the correct changes were made.

Last updated on 27 August 2024

"""

# Import libraries.
import os

"""STEP 1. Create a list of all the EcoLight files to be modified."""

os.chdir(r'data/setup_files')
the_list = []

for root, dirs, files in os.walk(r'data/setup_files'):
    for file in files:
        if file.endswith('.txt'):
            the_list.append(file)

"""STEP 2. Write the new EcoLight set-up files."""


def change_and_save_bottom_file(list_files, original_file_path, bottom_name, bottom_file):
    # Read the original file into a list of lines
    for i in list_files:
        with open(original_file_path + i, 'r') as original_fp:
            original_file_content = original_fp.readlines()
            id_string = original_file_content[2].strip()
            new_id = id_string.replace('coralbrown', bottom_name)
            new_file_name = new_id.replace(bottom_name, '')
            original_file_content[2] = new_id + '\n'  # Rename the output file
            original_file_content[61] = bottom_file + '\n'  # Specify the name of the benthic reflectance file

            directory_path = 'data/setup_files/setup_modified/Icorals'

            # Create the directory if it doesn't exist
            os.makedirs(directory_path, exist_ok=True)

            # Construct the file path
            path = os.path.join(directory_path, f'Icorals{new_file_name}_{bottom_name}.txt')

            with open(path, 'w') as fp:
                for item in original_file_content:
                    fp.write(item)
    return original_file_content


original_path = 'data/setup_files/'
# change_and_save_bottom_file(the_list, original_path, 'sand', 'sand.txt')

"""STEP 3. Check that the correct changes were made."""

# Check that only lines 2 and 61 were changed
# reading files
f1 = open('data/setup_files/Icorals_00_00_021_461_672_1006_coralbrown.txt', 'r')
f2 = open('data/setup_files/Icorals_00_00_021_461_672_1006_sand.txt', 'r')

f1_data = f1.readlines()
f2_data = f2.readlines()
num_lines = (len(f1_data))

for x in range(0, num_lines):
    # compare each line one by one
    if f1_data[x] != f2_data[x]:
        print("Difference detected - Line ", x, ":")
        print("\tFile 1:", f1_data[x], end='')
        print("\tFile 2:", f2_data[x], end='')

# close the files
f1.close()
f2.close()
