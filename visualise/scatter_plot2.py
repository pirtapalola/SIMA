import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Read the csv file containing the data into a pandas dataframe
data_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/Summary_Hyper/"
data = pd.read_csv(data_path + "CI_widths_summary.csv")

snr_level_list = []
for item in data['SNR']:
    snr_level_list.append(str(item))
data["SNR"] = snr_level_list

plt.figure(figsize=(5, 6))
colors = {'50': '#a3f7b5', '100': '#7fc8f8', '500': '#08a045'}
marker_styles = {'50': 'o', '100': 's', '500': 'X'}

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

"""
for snr_level in data['SNR'].unique():
    subset = data[data['SNR'] == snr_level]
    sns.scatterplot(subset, x="Parameter", y='CI_value', color=colors[snr_level], label=snr_level)"""

plt.ylabel('Frequency')
plt.legend(title='SNR')
plt.savefig(data_path + 'CI_widths.tiff')  # Save the figure as a tiff file
plt.show()
