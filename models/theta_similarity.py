"""

Compare a reference row with all the rows in a dataframe.
Filter a dataframe.

"""
# Import libraries
import pandas as pd
from sklearn.metrics import mean_squared_error

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

"""STEP 2. Filter the dataframe."""

# Specify the range for each column
column_ranges = {'phy': (0.005, 0.05),
                 'cdom': (0.1, 0.5),
                 'spm': (1.8, 2.2),
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
