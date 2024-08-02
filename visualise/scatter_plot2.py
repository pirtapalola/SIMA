import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Read the csv file containing the data into a pandas dataframe
data_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/"
data = pd.read_csv(data_path + "CI_widths_summary_resolution.csv")

snr_level_list = []
for item in data['SNR']:
    snr_level_list.append(str(item))
data["SNR"] = snr_level_list

plt.figure(figsize=(5, 6))
colors = {'50': '#a3f7b5', '100': '#7fc8f8', '500': '#08a045'}
marker_styles = {'50': 'o', '100': 's', '500': 'X'}

# colors = {'Hyper': '#7fc8f8', 'Multi': '#08a045'}
# marker_styles = {'Hyper': 'o', 'Multi': 's'}

# Plot with hue for colors and style for markers
sns.scatterplot(
    data=data,
    x="Parameter",
    y="CI_value",
    hue="SNR",
    style="SNR",
    palette=colors,
    markers=marker_styles,
    legend='full'
)

# Define the ticks and labels of the y-axis
y_ticks = np.arange(0, 15, 5)
plt.yticks(y_ticks)

plt.ylabel('Width of the 95% confidence interval')
# plt.legend().remove()
plt.legend(title='SNR', loc='upper left')
plt.savefig(data_path + 'CI_widths_noise.tiff')  # Save the figure as a tiff file
plt.show()
