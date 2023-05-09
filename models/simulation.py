"""This code creates the input files for the Hydrolight simulations"""
import pandas as pd


# Read the file
# Specify which rows to import
specific_rows = [6]

concentrations = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Hydrolight_setup/Icorals.txt',
                   sep=',', header=None, skiprows=lambda x: x not in specific_rows)
print(concentrations)