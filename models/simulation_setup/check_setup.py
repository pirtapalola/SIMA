"""

SIMULATION SET-UP I: Checking set-up files
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Check the EcoLight set-up files by identifying differences between two files line-by-line.

The simulated data is available in the associated Open Science Framework data repository:
Palola, P. (2024, August 26). SBI_marine_remote_sensing. osf.io/pcdgv

Last updated on 27 August 2024

"""

# Open the files
f1 = open("data/simulated_data/Icorals_00_032_016_671_799_1261_coralbrown.txt", "r")
f2 = open("data/simulated_data/Icorals_00_018_207_2198_528_1834_coralbrown.txt", "r")

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
