"""This code creates input files for HydroLight simulations"""

import pandas as pd
import itertools

# Open the file. Each line is saved as a string in a list.

with open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/Icorals.txt') as f:
    concentrations = [line for line in f.readlines()]

# Print the line that specifies water constituent concentrations.
# 1st element: water
# 2nd element: phytoplankton
# 3rd element: CDOM
# 4th element: SPM

# Define lists that contain the different concentrations of each water constituent.
water = [0]
phy = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]
cdom = [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5]
spm = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30]

# Create all the possible combinations of water constituent concentrations.
combinations = list(itertools.product(water, phy, cdom, spm))
print(len(combinations)) # print the number of combinations
combinations0 = combinations[0]

strcomb0 = ', '.join(str(n) for n in combinations0)
print(strcomb0)
str0 = strcomb0 + ', \n'

# Save the combinations in a csv file
df = pd.DataFrame(combinations, columns=['water', 'phy', 'cdom', 'spm'])
# print(df)
# df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/water_constituent_combinations.csv')

new_combination = concentrations
new_combination[6] = str0

# open file in write mode
with open(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/Icorals001.txt', 'w') as fp:
    for item in new_combination:
        # write each item on a new line
        fp.write(item)

# Check that only the 6th line was changed

# reading files
f1 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/Icorals001.txt', 'r')
f2 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/Icorals.txt', 'r')

f1_data = f1.readlines()
f2_data = f2.readlines()
num_lines = (len(f1_data))

for x in range(0, num_lines):
    # compare each line one by one
    if f1_data[x] != f2_data[x]:
        print("Line ", x, ":")
        print("\tFile 1:", f1_data[x], end='')
        print("\tFile 2:", f2_data[x], end='')

# close the files
f1.close()
f2.close()
