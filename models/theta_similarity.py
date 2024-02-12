"""

Calculate a similarity score between a reference row and each row in a dataframe.

"""

# Import libraries
import pandas as pd
from scipy.spatial.distance import euclidean

# Read the data
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
       'Dec2023_lognormal_priors/'
theta_df = pd.read_csv(path + "Ecolight_parameter_combinations.csv")
theta_df = theta_df.drop(['water', 'wind'], axis=1)
reference_theta = [0.1, 3.0, 0.1, 1.1]

# Calculate Euclidean distance between reference row and all rows in the DataFrame
distance_scores = [euclidean(reference_theta, row) for _, row in theta_df.iterrows()]

# Convert the distance scores to a DataFrame
distance_df = pd.DataFrame(distance_scores, columns=['Distance'], index=theta_df.index)

# Sort the DataFrame by distance in ascending order and get the top 10 rows
top_10_rows = distance_df.sort_values(by='Distance').head(10)

# Specify the range for each column
column_ranges = {'phy': (0.05, 0.3),
                 'cdom': (0.05, 0.5),
                 'spm': (2, 4),
                 'depth': (0.5, 2.0)}

# Iterate over rows and check if values are within the specified range for each column
selected_rows = []

for index, row in theta_df.iterrows():
    if all(column_ranges[col][0] <= row[col] <= column_ranges[col][1] for col in theta_df.columns):
        selected_rows.append(index)

# Display the selected rows
selected_df = theta_df.loc[selected_rows]
print(selected_df)
