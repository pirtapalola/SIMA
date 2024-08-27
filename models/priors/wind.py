"""

PRIORS I: Estimating the prior distribution for wind speed.
This code is part of the project "Simulation-based inference for marine remote sensing" by Palola et al.

Estimate the wind speed prior distribution using Copernicus data.

Last updated on 27 August 2024

"""

# import copernicus_marine_client as copernicus_marine
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, gamma, lognorm
# import seaborn as sns
import scipy
import torch
# from models.tools import fit_lognormal_torch
# from statsmodels.distributions.empirical_distribution import ECDF

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
wind_tahiti_2017_2022.to_csv("data/simulation_setup/wind_copernicus.csv")
"""

wind_df = pd.read_csv("data/simulation_setup/wind_copernicus.csv")

# Drop rows with NaN values.
wind_df = wind_df.dropna(subset=['wind_speed'])
maximum = wind_df['wind_speed'].max()
minimum = wind_df['wind_speed'].min()
print(minimum, maximum)

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


# Apply the function.
average_wind_speed = average_wind(dates, wind_df)

# Convert the data into a numpy array.
wind = average_wind_speed["wind_speed"].values
print("Mean wind speed: ", wind.mean())

size = len(wind)
x = np.linspace(0, 20, size)  # Use linspace to create x-values

# Create the histogram
h = plt.hist(wind, bins=100, density=True)  # Use density=True to get normalized probabilities

# Test fitting different distributions to the data
dist_names = ['norm', 'lognorm', 'gamma']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    params = dist.fit(wind)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    print(arg, loc, scale)
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
    else:
        pdf_fitted = dist.pdf(x, loc=loc, scale=scale)
    plt.plot(x, pdf_fitted, label=dist_name)
    # Set x-axis limits
    plt.xlim(0, 20)
    # Add labels and legend
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper right')


#
def fit_lognormal_moments(data):
    mu = torch.mean(torch.log(data))
    sigma = torch.std(torch.log(data))

    return mu.item(), sigma.item()


# Fit log-normal distribution to the data using method of moments
wind_tensor = torch.from_numpy(wind)
mu, sigma = fit_lognormal_moments(wind_tensor)

# Create PyTorch log-normal distribution with the fitted parameters
torch_lognormal_dist = torch.distributions.LogNormal(mu, sigma)

# Assuming you already have the fitted parameters for each distribution
params_norm = norm.fit(wind)
params_lognorm = lognorm.fit(wind)
params_gamma = gamma.fit(wind)

# Perform KS tests
ks_statistic_norm, ks_pvalue_norm = scipy.stats.kstest(wind, 'norm', params_norm)
ks_statistic_lognorm, ks_pvalue_lognorm = scipy.stats.kstest(wind, 'lognorm', params_lognorm)
ks_statistic_gamma, ks_pvalue_gamma = scipy.stats.kstest(wind, 'gamma', params_gamma)

print('KS Test Results:')
print('Normal Distribution: KS Statistic = {:.4f}, p-value = {:.4f}'.format(ks_statistic_norm, ks_pvalue_norm))
print('Lognormal Distribution: KS Statistic = {:.4f}, p-value = {:.4f}'.format(ks_statistic_lognorm, ks_pvalue_lognorm))
print('Gamma Distribution: KS Statistic = {:.4f}, p-value = {:.4f}'.format(ks_statistic_gamma, ks_pvalue_gamma))
print('Parameters of the lognormal distribution: ', 'mu = ', mu, ', sigma = ', sigma)
print('Number of data points: ', len(wind))

# Show the plot
plt.show()
