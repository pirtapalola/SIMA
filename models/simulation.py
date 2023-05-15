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

print(concentrations[6])

# Define lists that contain the different concentrations of each water constituent.
water = [0]
phy = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10]
cdom = [0.01, 0.1, 0.25, 0.5, 1, 2, 3, 5]
spm = [0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30]

# Create all the possible combinations of water constituent concentrations.
combinations = list(itertools.product(water, phy, cdom, spm))
print(len(combinations)) # print the number of combinations
print(combinations[0])
print(type(combinations[0]))
# Creating a string separator/delimiter
# Here it's a single space
st = ' '

# Using the Python join() function to convert the tuple to a string
st = st.join(combinations[0])
print(st)

# Save the combinations in a csv file
# df = pd.DataFrame(combinations, columns=['water', 'phy', 'cdom', 'spm'])
# df.to_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/water_constituent_combinations.csv')

new_combination = concentrations
print(new_combination)
new_combination[6] = combinations[0]
print(new_combination)

# open file in write mode
with open(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL/Icorals001.txt', 'w') as fp:
    for item in new_combination:
        # write each item on a new line
        fp.write("\n" % item)