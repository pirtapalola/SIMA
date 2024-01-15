# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create bars
bars_SE = ['>0-2km²', '>2-10km²', '>10-50km²', '>50-500km²', '>500-10,000km²', '>10,000-100,000km²', '>100,000km²']
x_SE = [15, 15, 13, 11, 7, 2, 4]
bars_SR = ['>0-1m', '>1-10m', '>10-100m', '>100-1000m', '>1000-5000m', '>5000-20,000m', '>20,000m']
x_SR = [3, 2, 4, 12, 5, 4, 3]
bars_TE = ['>0-1 day', '2-7 days', '1-4 weeks', '1-6 months', '6-12 months', '1-2 years', '2-5 years', '5-20 years', '20-50 years', '>50 years']
x_TE = [9, 8, 11, 4, 19, 6, 1, 4, 3, 2]
bars_TR = ['<1min', '>1-60min', '>1-12h', '>12-24h', '2-7 days', '1-4 weeks', '1-6 months', '6-12 months', '1-2 years', '2-5 years']
x_TR = [2, 16, 2, 6, 4, 4, 10, 8, 8, 1]

y_pos_S = np.arange(len(bars_SE))
y_pos_T = np.arange(len(bars_TE))

# Create subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot horizontal bars on each subplot
axs[0, 0].barh(y_pos_S, x_SE, color="#3a7ca5")
axs[0, 0].set_yticks(y_pos_S)
axs[0, 0].set_yticklabels(bars_SE)
axs[0, 0].set_title('Spatial extent')

axs[0, 1].barh(y_pos_S, x_SR, color="#3a7ca5")
axs[0, 1].set_yticks(y_pos_S)
axs[0, 1].set_yticklabels(bars_SR)
axs[0, 1].set_title('Spatial resolution')

axs[1, 0].barh(-y_pos_T, x_TE, color="#38a3a5")
axs[1, 0].set_yticks(-y_pos_T)
axs[1, 0].set_yticklabels(bars_TE)
axs[1, 0].set_title('Temporal extent')

axs[1, 1].barh(-y_pos_T, x_TR, color="#38a3a5")
axs[1, 1].set_yticks(-y_pos_T)
axs[1, 1].set_yticklabels(bars_TR)
axs[1, 1].set_title('Temporal resolution')

# Set the x-axis ticks format to not show decimals for all subplots
for ax in axs.flat:
    ax.set_xticks(np.arange(0, max(x_TE)+5, 5))
    ax.set_xticklabels(['{:,.0f}'.format(val) for val in np.arange(0, max(x_TE)+5, 5)])
    ax.set_xlabel('Number of studies')

# Adjust the layout
plt.tight_layout()

# Show graphic
plt.show()