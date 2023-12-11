"""

Plot histograms of the prior distributions.

Last updated on 11 December 2023 by Pirta Palola

"""

import matplotlib.pyplot as plt

# Plot the histogram of the prior distribution
plt.hist(prior_chl.numpy(), bins=3000, density=True, color='blue', alpha=0.7)
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Density')
plt.xlim(0, 20)
plt.show()
