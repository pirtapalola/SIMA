
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the csv file containing the inputs of each of the HydroLight simulation runs
hydrolight_input = pd.read_csv('C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/'
                               'Jan2024_lognormal_priors/Ecolight_parameter_combinations.csv')
hydrolight_input = hydrolight_input.drop(columns="water")  # Remove the "water" column.
samples_phy = hydrolight_input["phy"]
samples_cdom = hydrolight_input["cdom"]
samples_nap = hydrolight_input["spm"]
samples_wind = hydrolight_input["wind"]
samples_depth = hydrolight_input["depth"]

samples_phy = [i for i in samples_phy if i != 0]
samples_cdom = [i for i in samples_cdom if i != 0]
samples_nap = [i for i in samples_nap if i != 0]
samples_wind = [i for i in samples_wind if i != 0]
samples_depth = [i for i in samples_depth if i != 0]


samples_phy = np.log(samples_phy)
samples_cdom = np.log(samples_cdom)
samples_nap = np.log(samples_nap)
samples_wind = np.log(samples_wind)
samples_depth = np.log(samples_depth)

# Create a figure with four subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))

# Plot the PDF and histogram of prior1 in the first subplot
axes[0, 0].hist(samples_phy, bins=100, alpha=0.5, color='#76c893', label='Samples (n = 30,000)')
axes[0, 0].legend()
axes[0, 0].set_title('Phytoplankton')
axes[0, 0].set_xlim(left=0, right=2)  # Set x-axis limits
axes[0, 0].set_ylim(bottom=0, top=5000)  # Set y-axis limits
axes[0, 0].set_xlabel('Concentration (mg/$\mathregular{m^3}$)')
axes[0, 0].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior2 in the second subplot
axes[0, 1].hist(samples_cdom, bins=100, alpha=0.5, color='#76c893', label='Samples (n = 30,000)')
axes[0, 1].legend()
axes[0, 1].set_title('CDOM')
axes[0, 1].set_xlim(left=0, right=1)  # Set x-axis limits
axes[0, 1].set_ylim(bottom=0, top=5000)  # Set y-axis limits
axes[0, 1].set_xlabel('Absorption $\mathregular{m^-1}$ at 440 nm)')
axes[0, 1].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior3 in the third subplot
axes[1, 0].hist(samples_nap, bins=100, alpha=0.5, color='#76c893', label='Samples (n = 30,000)')
axes[1, 0].legend()
axes[1, 0].set_title('NAP')
axes[1, 0].set_xlim(left=0, right=5)  # Set x-axis limits
axes[1, 0].set_ylim(bottom=0, top=5000)  # Set y-axis limits
axes[1, 0].set_xlabel('Concentration (g/$\mathregular{m^3}$)')
axes[1, 0].set_ylabel('Probability Density')

# Plot the PDF and histogram of prior4 in the fourth subplot
axes[1, 1].hist(samples_wind, bins=100, alpha=0.5, color='#76c893', label='Samples (n = 30,000)')
axes[1, 1].legend()
axes[1, 1].set_title('Wind')
axes[1, 1].set_xlim(left=0, right=22)  # Set x-axis limits
axes[1, 1].set_ylim(bottom=0, top=5000)  # Set y-axis limits
axes[1, 1].set_xlabel('Wind speed (m/s)')
axes[1, 1].set_ylabel('Probability Density')

# Plot histogram of prior5 in the fifth subplot
axes[2, 0].hist(samples_depth, bins=100, alpha=0.5, color='#76c893', label='Samples (n = 30,000)')
axes[2, 0].legend()
axes[2, 0].set_title('Depth')
axes[2, 0].set_xlim(left=0, right=20)  # Set x-axis limits
axes[2, 0].set_ylim(bottom=0, top=5000)  # Set y-axis limits
axes[2, 0].set_xlabel('Depth (m)')
axes[2, 0].set_ylabel('Probability Density')

fig.tight_layout()
plt.show()
