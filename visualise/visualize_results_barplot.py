import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Read the csv file containing the data into a pandas dataframe
data_path = "C:/Users/kell5379/Documents/Chapter2_May2024/Final/Results1_constrained/"
df = pd.read_csv(data_path + "results_summary_all.csv")

# Only keep the 100 SNR
df = df[df.SNR != "50"]
df = df[df.SNR != "500"]

# Set the order of the parameters for consistent plotting
parameter_order = ['Phytoplankton', 'Minerals', 'Wind', 'Depth']

# Set font size
font = {'size': 12}
plt.rc('font', **font)

# Set the color palette
colors = ["#00b4d8", "green"]

# Create a barplot with confidence intervals for each model and parameter
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.barplot(x='Parameter', y='CI_width', hue='Resolution', data=df, order=parameter_order,
            errorbar=None, palette=colors, alpha=0.5)

# Add labels and title
plt.xlabel('Parameter')
plt.ylabel('Width of the 95% confidence interval')
plt.legend(title='Spectral resolution')

# Show the plot
plt.show()
