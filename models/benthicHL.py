import pandas as pd

tetiaroa_data = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/avg_coral_tetiaroa.csv')
wavelength_data = tetiaroa_data['wavelength']
reflectance_data = tetiaroa_data['reflectance']
wavelength = [str(i) for i in wavelength_data]
reflectance = [str(i) for i in reflectance_data]

# open file in write mode
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/HL_benthic/avg_coral.txt'


# hydrolight_file[65] = r'D:\HE53\data\User\microplastics\MPzdata.txt' + '\n'

def write_wavelength_file(wavelength_list):
    with open(path, 'w') as fp:
        for item in wavelength_list:
            fp.write(' ' + item + '   ' + '\n')


write_wavelength_file(wavelength)


with open(path) as fp:
    lines = fp.read().splitlines()
with open(path, "w") as fp:
    for line in lines:
        for value in reflectance:
            print(line + value, file=fp)

# print(data)
# print(tetiaroa_data)
