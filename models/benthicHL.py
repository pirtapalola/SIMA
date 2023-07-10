import pandas as pd

PATH = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/'

data = pd.read_csv(PATH + 'avg_coral.txt', sep=' ', skiprows=10, header=None)
tetiaroa_data = pd.read_csv(PATH + 'avg_coral_tetiaroa.csv')


default_coral_reflectance = data[4]
print(default_coral_reflectance)
print(tetiaroa_data)

default_coral_reflectance.to_csv(PATH + 'default_coral_reflectance.csv')
