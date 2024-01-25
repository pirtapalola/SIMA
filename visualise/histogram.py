"""

Plot histograms of the prior distributions.

Last updated on 11 December 2023 by Pirta Palola

"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from models.tools import TruncatedLogNormal

# Import the data.
path = 'C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/' \
       'Jan2024_lognormal_priors/priors/summary_priors.csv'
df = pd.read_csv(path)

# Save the priors as lists.
chl = df['chl']
cdom = df['cdom']
nap = df['nap']
wind = df['wind']
depth = df['depth']

# Plot the data in a histogram.
plt.hist(wind, bins=10000, density=True, color='#69b3a2', alpha=0.7)

upper_bound = 20
prior = TruncatedLogNormal(loc=0, scale=5, upper_bound=upper_bound)

x_values = np.linspace(0.01, upper_bound, 1000)
pdf_values = prior.pdf(torch.tensor(x_values))
# plt.plot(x_values, pdf_values, label='Probability density function')

plt.xlabel('Wind speed (m/s)')
plt.ylabel('Probability density')
plt.title(r'$\mu=1.85$, $\sigma=0.33$')

plt.xlim(0, upper_bound)
plt.show()
