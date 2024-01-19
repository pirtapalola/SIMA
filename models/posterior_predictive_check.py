"""

Conduct a posterior predictive check.

Last updated on 19 January 2024 by Pirta Palola

"""

# Import libraries
import pickle
from sbi.analysis import pairplot

"""STEP 1. Load the posterior."""

# Load the posterior
with open("C:/Users/pirtapalola/Documents/DPhil/Chapter2/Methods/Methods_Ecolight/"
          "Dec2023_lognormal_priors/loaded_posterior.pkl", "rb") as handle:
    loaded_posterior = pickle.load(handle)

# A PPC is performed after we trained or neural posterior
loaded_posterior.set_default_x(x_o)

# We draw theta samples from the posterior. This part is not in the scope of SBI
posterior_samples = loaded_posterior.sample((5_000,))

# We use posterior theta samples to generate x data
x_pp = simulator(posterior_samples)

# We verify if the observed data falls within the support of the generated data
_ = pairplot(
    samples=x_pp,
    points=x_o
)
