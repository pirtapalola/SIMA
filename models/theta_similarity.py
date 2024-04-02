"""

Compare a reference row with all the rows in a dataframe.
Filter a dataframe.

"""
# Import libraries
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

"""STEP 1. Mean squared error."""

# Read the data
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
       'Jan2024_lognormal_priors/'
theta_df = pd.read_csv(path + "Ecolight_parameter_combinations.csv")
theta_df = theta_df.drop(['water', "wind"], axis=1)
"""
# Define mean theta
mean_theta = [
    0.010,
    0.277,
    2.065,
    1.832,
    0.603]

phy_original = (np.exp(0.010))-1
cdom_original = (np.exp(0.277))-1
nap_original = (np.exp(2.065))-1
wind_original = np.exp(1.832)

print(phy_original, cdom_original, nap_original, wind_original)


# Calculate MSE between the mean theta and all rows in the DataFrame
MSE_values = [mean_squared_error(mean_theta, row) for _, row in theta_df.iterrows()]

# Save the MSE values in a dataframe
MSE_df = pd.DataFrame(MSE_values, columns=['MSE'], index=theta_df.index)

# Sort the DataFrame by distance in ascending order and get the top 10 rows
top_10_rows = MSE_df.sort_values(by='MSE', ascending=True).head(10)
print(top_10_rows)"""

"""
The results were:
324    0.271041
22020  0.376209
10854  0.433137
24327  0.449093
11928  0.469809
26537  0.472833
24942  0.595517
16007  0.607061
8865   0.652893
24150  0.656337
"""

"""STEP 2. Filter the dataframe.

0.10896802
0.38263
7.102253
6.5983033
0.61579853

"""

# Specify the range for each column
column_ranges = {'phy': (0.01, 0.4),
                 'cdom': (0.1, 0.7),
                 'spm': (6.5, 7.5),
                 'depth': (0.5, 0.9)}

# Iterate over rows and check if values are within the specified range for each column
selected_rows = []

for index, row in theta_df.iterrows():
    if all(column_ranges[col][0] <= row[col] <= column_ranges[col][1] for col in theta_df.columns):
        selected_rows.append(index)

# Display the selected rows
selected_df = theta_df.loc[selected_rows]
print(selected_df)
# print(len(selected_rows))
