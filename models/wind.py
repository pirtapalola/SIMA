import copernicus_marine_client as copernicus_marine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions

# Set parameters
data_request = {
   "dataset_id": "cmems_obs-wind_glo_phy_nrt_l3-hy2b-hscat-asc-0.25deg_P1D-i",
   "longitude": [-150, -149],
   "latitude": [-18, -17],
   "time": ["2017-01-01", "2022-12-31"],
   "variables": ["wind_speed"]
}

"""
# Read the dataframe
wind_tahiti_2017_2022 = copernicus_marine.read_dataframe(
    dataset_id=data_request["dataset_id"],
    minimum_longitude=data_request["longitude"][0],
    maximum_longitude=data_request["longitude"][1],
    minimum_latitude=data_request["latitude"][0],
    maximum_latitude=data_request["latitude"][1],
    start_datetime=data_request["time"][0],
    end_datetime=data_request["time"][1],
    variables=data_request["variables"]
)

# Print and save csv
print(wind_tahiti_2017_2022)
wind_tahiti_2017_2022.to_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/"
                             "Methods/Methods_Ecolight/wind/wind_tahiti_2017_2022.csv")
"""

wind_df = pd.read_csv("C:/Users/pirtapalola/Documents/DPhil/Chapter2/"
                      "Methods/Methods_Ecolight/wind/wind_tahiti_2017_2022.csv")

# Drop rows with NaN values.
wind_df = wind_df.dropna(subset=['wind_speed'])

# Create a list of dates for which there is data.
dates = wind_df['time'].unique()

# Write a function that calculates the average wind speed at each date.


def average_wind(list_dates, dataframe_to_split):
    datalist = []
    datelist = []
    new_df = pd.DataFrame()
    for i in range(0, len(list_dates)):
        data = dataframe_to_split.loc[dataframe_to_split['time'] == list_dates[i]]  # Split the data by date
        average_w = data['wind_speed'].mean()  # Calculate average wind speed at each date
        datalist.append(average_w)  # Add the date to a list
        datelist.append(list_dates[i])
    new_df['time'] = datelist
    new_df['wind_speed'] = datalist
    return new_df


average_wind_speed = average_wind(dates, wind_df)
print(average_wind_speed)

# plt.hist(average_wind_speed['wind_speed'], bins=100)
# plt.show()

sns.set_style('white')
sns.set_context("paper", font_scale=2)
sns.displot(data=average_wind_speed['wind_speed'], x="Wind speed", kind="hist", bins=100, aspect=1.5)

# Convert the data into a numpy array
wind_speed = average_wind_speed['wind_speed'].values

f = Fitter(height,
           distributions=['gamma',
                          'lognorm',
                          "beta",
                          "burr",
                          "norm"])
f.fit()
f.summary()
