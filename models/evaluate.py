"""

Assess the performance of the inference scheme.
STEP 1. Sample from the posterior.
STEP 2. Read the ground-truth data.
STEP 3. Define functions to assess inference performance.
STEP 4. Apply the functions.

Last updated on 2 June 2024 by Pirta Palola

"""

# Import libraries
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import gaussian_kde
import pickle
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""STEP 1. Sample from the posterior."""

# Define parameter of interest (0 = phy, 1 = cdom, 2 = spm, 3 = wind, 4 = depth)
param_index = 0

# Define sample IDs 'ONE05', 'RIM03', 'RIM04', 'RIM05'
sample_id_list = []
for i in range(1, 1001):
    sample_id_list.append(str(i))

print(len(sample_id_list))


# Load the posterior
with open("C:/Users/kell5379/Documents/Chapter2_May2024/Final/Trained_nn/1000SNR/"
          "Loaded_posteriors/loaded_posterior29.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# Read the csv file containing the observation data
observation_path = ('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Evaluation_data/'
                    'simulated_reflectance_1000SNR_noise_sbc_transposed.csv')
# obs_file = 'hp_field_1000SNR.csv'
obs_df = pd.read_csv(observation_path)  # + obs_file


# Define a function to sample from the posterior
def posterior_sampling(sample_id, observation_dataframe):
    x_obs = observation_dataframe[sample_id].to_list()
    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    # modified_data = torch.cat((samples[:, :1], samples[:, 2:]), dim=1)
    posterior_samples_array = samples.numpy()  # Convert to NumPy array
    return posterior_samples_array


# Create an empty list
posterior_list = []

# Add the posterior samples of each field site as elements into the list
for item in sample_id_list:
    samples_i = posterior_sampling(item, obs_df)
    posterior_list.append(samples_i)

"""STEP 2. Read the ground-truth data."""

# Read the csv file containing the observation data
# observation_path = 'C:/Users/kell5379/Documents/Chapter2_May2024/Final/Field_data/'
# param_file = 'parameters_TET22.csv'
# obs_parameters = pd.read_csv(observation_path + param_file)


# Read the csv file containing the inputs of each of the EcoLight simulation runs
obs_parameters = pd.read_csv('C:/Users/kell5379/Documents/Chapter2_May2024/Final/Evaluation_data/'
                             'Ecolight_parameter_combinations_sbc.csv')
unique_ids = obs_parameters["unique_ID"]
unique_ids = [str(n) for n in unique_ids]

# Add a constant to avoid issues with the log-transformation of small values
constant = 1.0
samples_phy = [i+constant for i in obs_parameters["phy"]]
samples_cdom = [i+constant for i in obs_parameters["cdom"]]
samples_nap = [i+constant for i in obs_parameters["spm"]]
samples_wind = obs_parameters["wind"]
samples_depth = obs_parameters["depth"]

# Conduct the log-transformation
samples_phy = np.log(samples_phy)
samples_phy = [round(item, 3) for item in samples_phy]
samples_cdom = np.log(samples_cdom)
samples_cdom = [round(item, 3) for item in samples_cdom]
samples_nap = np.log(samples_nap)
samples_nap = [round(item, 3) for item in samples_nap]
samples_wind = np.log(samples_wind)
samples_wind = [round(item, 3) for item in samples_wind]

# Save the transformed data in a dataframe
transformed_dictionary = {"unique_ID": unique_ids,
                          "phy": samples_phy, "cdom": samples_cdom, "spm": samples_nap,
                          "wind": samples_wind, "depth": samples_depth}

transformed_theta = pd.DataFrame(data=transformed_dictionary)


# Function to define ground-truth parameters
def groundtruth(sample_id, theta_dataframe):
    theta_obs = theta_dataframe.loc[theta_dataframe['unique_ID'] == sample_id]
    theta_obs = theta_obs.drop(columns=["unique_ID"])
    theta_obs = theta_obs.iloc[0].to_list()
    theta_obs_array = np.array(theta_obs)  # Convert to NumPy array
    return theta_obs_array


# Create an empty list
gt_list = []

# Add the posterior samples of each field site as elements into the list
for item in sample_id_list:
    gt_i = groundtruth(item, transformed_theta)
    gt_list.append(gt_i)

"""STEP 3. Define functions to assess inference performance."""


# Measure how often the true parameter value (within a % error margin)
# falls within the credible intervals of the posterior distributions
def coverage_probability(post_samples_list, true_values, error_margin=0.0):
    coverage_counts = 0
    for post_samples, true_value in zip(post_samples_list, true_values):
        theta_intervals = np.percentile(post_samples, [2.5, 97.5], axis=0)
        lower_bound = theta_intervals[0]
        upper_bound = theta_intervals[1]
        true_value_lower = true_value * (1 - error_margin)
        true_value_upper = true_value * (1 + error_margin)
        if (true_value_lower >= lower_bound and true_value_lower <= upper_bound) or \
           (true_value_upper >= lower_bound and true_value_upper <= upper_bound) or \
           (true_value >= lower_bound and true_value <= upper_bound):
            coverage_counts += 1
    return coverage_counts / len(true_values)


# Measure the distance between the estimated posterior distributions and the true parameter value,
# considering the entire shape of the distributions
def wasserstein(post_samples, true_value):
    return wasserstein_distance(post_samples, [true_value])


# The average squared difference between the true parameter values and samples from the posterior distribution
def pmse(post_samples_list, true_values):
    return np.mean([np.mean((post_samples - true_value) ** 2)
                    for post_samples, true_value in zip(post_samples_list, true_values)])


# Evaluate the posterior distributions based on the log probability of the true parameter values
# Higher log scores correspond to better performance
def log_score(post_samples_list, true_values_list):
    log_probs = []
    for post_samples, true_value in zip(post_samples_list, true_values_list):
        kde = gaussian_kde(post_samples)
        log_probs.append(kde.logpdf(true_value))
    return np.mean(log_probs)


# Calculate log probability
def evaluate_log_probability(sample_id, observation_dataframe):
    x_obs = observation_dataframe[sample_id].to_list()
    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    log_probability = loaded_posterior.log_prob(samples, x=x_obs)
    log_prob_np = log_probability.numpy()  # Convert to Numpy array
    return log_prob_np


# Calculate mean
def calculate_mean(sample_id, observation_dataframe):
    x_obs = observation_dataframe[sample_id].to_list()
    x_obs = torch.tensor(x_obs, dtype=torch.float32)
    samples = loaded_posterior.sample((1000,), x=x_obs)  # Sample from the posterior p(θ|x)
    theta_means = torch.mean(samples, dim=0)
    theta_exp = theta_means
    for i in range(4):  # Apply an exponential transformation to the first 4 theta parameters
        theta_exp[i] = np.exp(theta_means[i])
    for i in range(3):  # Remove the constant from the first 3 theta parameters
        theta_exp[i] = theta_exp[i] - constant
    return theta_exp


"""STEP 4. Apply the functions."""


# From the ground-truth array, access each parameter
def gt_per_parameter(gt_theta_index):
    empty_list = []  # Create an empty list
    for i in range(len(sample_id_list)):
        gt_array = gt_list[i]  # Loop through each sampling site
        theta_gt = gt_array[gt_theta_index]  # Extract a single theta parameter value
        empty_list.append(theta_gt)  # Append the extracted value to the list
    return empty_list  # The list contains the values of the selected theta parameter at each of the sampling sites


# From the dataset containing the posterior samples, access the samples associated with each parameter
def posterior_per_parameter(theta_index):
    empty_l = []
    for i in range(len(sample_id_list)):
        posterior_samples = posterior_list[i]  # Loop through each sampling site
        # Extract posterior samples corresponding to the specified theta parameter
        theta_posterior = [row[theta_index] for row in posterior_samples]
        empty_l.append(theta_posterior)
    return empty_l


# Apply the mean calculation function
mean_list = []
for item in sample_id_list:
    mean_i = calculate_mean(item, obs_df)
    mean_list.append(mean_i)


# From the dataset containing the mean values, access the values associated with each parameter
def mean_per_parameter(theta_index):
    empty_l = []
    for i in range(len(sample_id_list)):
        mean_values = mean_list[i]  # Loop through each sampling site
        mean_values = mean_values.numpy()  # Convert to NumPy array
        # Extract posterior samples corresponding to the specified theta parameter
        mean_value = mean_values[theta_index]
        empty_l.append(mean_value)
    return empty_l


gt = gt_per_parameter(param_index)
# print("GT: ", gt)
post = posterior_per_parameter(param_index)
# print("Post: ", post)
mean_estimate = mean_per_parameter(param_index)

print("Length of GT data: ", len(gt))
print("Length of mean data: ", len(mean_estimate))

# Calculate RMSE
y_actual = np.array(gt)
print("Length of y_actual: ", len(y_actual))
y_predicted = np.array(mean_estimate)
print("Length of y_predicted: ", len(y_predicted))

MSE = np.square(np.subtract(y_actual, y_predicted)).mean()
RMSE = math.sqrt(MSE)

# Apply the coverage probability evaluation function
coverage = coverage_probability(post, gt)

# Print the results
print(f"Coverage Probability: {coverage}")
# print(f"R squared: {MSE}")
# print(f"RMSE: {RMSE}")

mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_actual, y_predicted)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_actual, y_predicted, alpha=0.5)
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.title('Observed vs Predicted')
plt.show()

# Line Plot
plt.figure(figsize=(10, 6))
plt.plot(y_actual, label='Observed')
plt.plot(y_predicted, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Observed and Predicted Values')
plt.show()

# Residual Plot
residuals = y_actual - y_predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_predicted, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(y_actual, color='blue', kde=True, label='Observed', stat='density')
sns.histplot(y_predicted, color='orange', kde=True, label='Predicted', stat='density')
plt.legend()
plt.title('Distribution of Observed and Predicted Values')
plt.show()

# Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(y_actual, color='blue', label='Observed')
sns.kdeplot(y_predicted, color='orange', label='Predicted')
plt.legend()
plt.title('Density Plot of Observed and Predicted Values')
plt.show()
