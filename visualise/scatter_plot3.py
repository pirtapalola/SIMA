import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read the csv file containing the data into a pandas dataframe
data_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/"
data = pd.read_csv(data_path + "results_summary_all.csv")

snr_level_list = []
for item in data['SNR']:
    snr_level_list.append(str(item))
data["SNR"] = snr_level_list

data1 = data[data.Resolution != "Multispectral"]

data = data[data.SNR != "50"]
data = data[data.SNR != "500"]
# data = data[data.Resolution != "Multispectral"]

plt.figure(figsize=(5, 15))
# colors = {'50': '#a3f7b5', '100': '#7fc8f8', '500': '#08a045'}
# marker_styles = {'50': 'o', '100': 's', '500': 'X'}
# colors = {'Hyperspectral': '#7fc8f8', 'Multispectral': '#08a045'}
# marker_styles = {'Hyperspectral': 'o', 'Multispectral': 's'}

# Create a boxplot
# sns.barplot(data=data1, x='Parameter', y='CI_width', hue='SNR', errorbar=None)

# Create a boxplot
sns.barplot(data=data, x='Parameter', y='CI_width', hue='Resolution', errorbar=None)

# Define the ticks and labels of the y-axis
y_ticks = np.arange(0, 20, 5)
plt.yticks(y_ticks)

plt.ylabel('Width of the 95% confidence interval')
# plt.legend().remove()
plt.legend(title='Spectral resolution', loc='upper left')
# plt.savefig(data_path + 'CI_widths_resolution.tiff')  # Save the figure as a tiff file
plt.show()
