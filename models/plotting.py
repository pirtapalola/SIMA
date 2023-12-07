import matplotlib.pyplot as plt
import torch
from scipy.stats import lognorm

def prior_lognormal(loc, scale):
    prior_lognormal_dist = torch.distributions.log_normal.LogNormal(torch.tensor([loc]), torch.tensor([scale]))
    input_data = prior_lognormal_dist.sample((5000,))
    return input_data


# Generate samples from the lognormal distribution using PyTorch
prior_wind = torch.from_numpy(lognorm.rvs(s=0.13295081858033553, loc=-9.08382565107291,
                                             scale=15.666340917893361, size=5000)).float()

# Generate the prior distributions
prior_chl = prior_lognormal(0.4, 1.6)
prior_cdom = prior_lognormal(0.2, 1.6)
prior_spm = prior_lognormal(1.3, 1.1)
#prior_wind = prior_lognormal(2.75, 0.13)

# Plot the histogram of the prior distribution
plt.hist(prior_wind.numpy(), bins=5000, density=True, color='blue', alpha=0.7)
plt.xlabel('Wind speed')
plt.ylabel('Density')
plt.xlim(0, 50)
plt.show()
