# Import libraries
import pandas as pd
from io import StringIO
import os

os.chdir(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
         r'check_simulation_Feb2024/test4/simulations')
the_list = []

# Create a list that contains the paths of all the csv files in a folder
for root, dirs, files in os.walk(r'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/'
                                 r'Methods_Ecolight/check_simulation_Feb2024/test4/simulations'):
    for file in files:
        if file.endswith('.csv'):
            the_list.append(file)


# A function to process a single file
def process_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        lines = text.strip().split('\n')[273:335]  # Extract lines
        data = "\n".join(lines)  # Join the lines and create a StringIO object
        data_io = StringIO(data)
        df = pd.read_csv(data_io, sep=r'\s+', header=None)  # Read the data into a pandas DataFrame
        df = df.T  # Transpose the DataFrame to get the desired format
        df.columns = df.iloc[0]  # Set the first row as the header
        df = df.iloc[:3]
        df = df.drop([0, 2])
    return df


empty_list = []
for file in range(len(files)):
    data = process_file(files[file])
    empty_list.append(data.iloc[0])

print(empty_list)
dataframe = pd.DataFrame(empty_list)
print(dataframe)

dataframe.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/"
                 "Methods_Ecolight/check_simulation_Feb2024/test4/test4.csv")
