"""

Plot histograms of the prior distributions.

Last updated on 11 December 2023 by Pirta Palola

"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import scipy.stats

# Import the data.
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/priors/summary_priors.csv'
df = pd.read_csv(path)

# Save the priors as lists.
chl = df['chl']
cdom = df['cdom']
nap = df['nap']
wind = df['wind']
depth = df['depth']

# Plot the data in a histogram.
plt.hist(depth, bins=3000, density=True, color='#69b3a2', alpha=0.7)

#fig, ax = plt.subplots()

# the histogram of the data
#n, bins, patches = ax.hist(chl, 3000, density=True, color='#69b3a2', alpha=0.7)
#ax.plot(bins)
plt.xlabel('Depth (m)')
plt.ylabel('Probability density')
#plt.set_title(r'$\mu=0.4$, $\sigma=1.6$')

plt.xlim(0, 20)
plt.show()
