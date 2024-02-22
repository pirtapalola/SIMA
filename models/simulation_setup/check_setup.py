# Check set-up files

# reading files
f1 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
          'Jan2024_lognormal_priors/setup/Icorals_00_032_016_671_799_1261_coralbrown.txt', 'r')
f2 = open('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
          'Jan2024_lognormal_priors/setup/Icorals_00_018_207_2198_528_1834_coralbrown.txt', 'r')

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
