"""

Change the bottom reflectance file name in the Hydrolight setup files.

"""

# Import libraries.
import os

"""STEP 1. Create a list of all the Hydrolight files to be modified."""

os.chdir(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup')
the_list = []

for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup'):
    for file in files:
        if file.endswith('.txt'):
            the_list.append(file)

"""STEP 2. Write the new Ecolight set-up files."""


def change_and_save_bottom_file(original_file_path, bottom_name, bottom_file):
    # Read the original file into a list of lines
    with open(original_file_path, 'r') as original_fp:
        original_file_content = original_fp.readlines()

    # Modify the desired lines
    id_string = original_file_content[2].strip()
    new_id = id_string.replace('coralbrown', bottom_name)
    original_file_content[2] = new_id + '\n'  # Rename the output file
    original_file_content[61] = bottom_file + '\n'  # Specify the name of the benthic reflectance file

    # Create a new file path based on the modifications
    new_file_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup/setup_sand/' \
                    + new_id + '.txt'

    # Write the modified content to the new file
    with open(new_file_path, 'w') as new_fp:
        new_fp.writelines(original_file_content)

    return new_file_path


original_path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup/'
change_and_save_bottom_file(original_path, 'sand', 'sand.txt')

"""STEP 3. Check that the correct changes were made."""

# Check that only line 61 was changed

# reading files
f1 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
          'setup/sand/Icorals_00_0001_1361_049_509_1927_sand.txt', 'r')
f2 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
          'final_setup/Icorals_final.txt', 'r')

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
