"""

Change the bottom reflectance file name in the Hydrolight setup files.

"""

"""STEP 4. Write the new Ecolight set-up files."""


def change_bottom_file(hydrolight_file, bottom_name, bottom_file):
    id_string = hydrolight_file[2]
    hydrolight_file[2] = bottom_name + id_string + '\n'  # rename the output file
    hydrolight_file[61] = bottom_file + '\n'  # specify the benthic reflectance
    # open file in write mode
    path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup/Icorals'\
           + id_string + '_' + bottom_name + '.txt'
    with open(path, 'w') as fp:
        for item in hydrolight_file:
            fp.write(item)
    return hydrolight_file


# Check that only line 61 was changed

# reading files
f1 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/setup/Icorals.txt', 'r')
f2 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/final_setup/Icorals_final.txt', 'r')

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
