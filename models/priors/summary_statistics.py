"""

Calculate summary statistics from samples drawn from the prior distributions.

"""

# Import libraries
import pandas as pd
import statistics
import numpy as np

# Specify the path
path = "C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/Jan2024_lognormal_priors/priors/"

# Read the csv files
chl_samples = pd.read_csv(path + "chl_prior.csv")
cdom_samples = pd.read_csv(path + "cdom_prior.csv")
spm_samples = pd.read_csv(path + "spm_prior.csv")

# Create lists
chl_list = chl_samples["0"]
cdom_list = cdom_samples["0"]
spm_list = spm_samples["0"]

chl_list = chl_list.to_list()
cdom_list = cdom_list.to_list()
spm_list = spm_list.to_list()

# Calculate summary statistics
print(chl_list)
print(len(chl_list), len(cdom_list), len(spm_list))

stats = ["Means", "Standard deviation", "Minimum", "Maximum", "10th percentile", "33rd percentile", "90th percentile"]
means = statistics.mean(chl_list), statistics.mean(cdom_list), statistics.mean(spm_list)
std_list = statistics.stdev(chl_list), statistics.stdev(cdom_list), statistics.stdev(spm_list)
medians = statistics.median(chl_list), statistics.median(cdom_list), statistics.median(spm_list)
percentile10 = np.percentile(chl_list, 10), np.percentile(cdom_list, 10), np.percentile(spm_list, 10)
percentile33 = np.percentile(chl_list, 33), np.percentile(cdom_list, 33), np.percentile(spm_list, 33)
percentile90 = np.percentile(chl_list, 90), np.percentile(cdom_list, 90), np.percentile(spm_list, 90)
min_list = min(chl_list), min(cdom_list), min(spm_list)
max_list = max(chl_list), max(cdom_list), max(spm_list)

# Print the summary statistics
print("Mean: ", means)
print("Median: ", medians)
print("10th percentile: ", percentile10)
print("33rd percentile: ", percentile33)
print("90th percentile: ", percentile90)
print("Min: ", min_list)
print("Max: ", max_list)

# Save the summary statistics into a csv file
summary_stats = pd.DataFrame()
summary_stats["Mean"] = means
summary_stats["Standard deviation"] = std_list
summary_stats["Median"] = medians
summary_stats["Minimum"] = min_list
summary_stats["Maximum"] = max_list
summary_stats["10th percentile"] = percentile10
summary_stats["33rd percentile"] = percentile33
summary_stats["90th percentile"] = percentile90
#summary_stats.to_csv(path + "prior_summary_stats.csv")
#print(summary_stats)
