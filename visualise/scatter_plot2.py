import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read the csv file containing the data into a pandas dataframe
data_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/Summary_Hyper/"
data = pd.read_csv(data_path + "CI_intervals_summary.csv")
snr_level_list = []
for item in data['SNR']:
    snr_level_list.append(str(item))
data["SNR"] = snr_level_list

plt.figure(figsize=(4, 6))
colors = {'50': '#a3f7b5', '100': '#7fc8f8', '500': '#08a045'}
for snr_level in data['SNR'].unique():
    subset = data[data['SNR'] == snr_level]
    sns.scatterplot(subset, x="Parameter", y='CI_value', color=colors[snr_level], label=snr_level)
# sns.scatterplot(data=data, x='Parameter', y='CI_value', hue='SNR', style='SNR')
plt.ylabel('Frequency')
plt.show()

# sns.pairplot(data, hue='Parameter')  # or hue='Parameter'

# Define distinct colors and opacity levels
colors = {'Phy': '#a8dadc', 'Min': '#6096ba', 'Wind': '#a3cef1', 'Depth': '#1d3557'}
alpha = 0.9  # Adjust this value to change opacity

plt.figure(figsize=(10, 6))
# sns.histplot(data=data, x="CI_value", hue='Parameter', bins=30)
for param in data['Parameter'].unique():
    subset = data[data['Parameter'] == param]
    sns.histplot(subset, x="CI_value", color=colors[param], alpha=alpha, label=param, kde=False, bins=20)
