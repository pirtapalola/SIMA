"""

Plot pdf for a Truncated LogNormal distribution.

"""

# Import libraries
import numpy as np
import matplotlib as plt
import torch
from models.tools import TruncatedLogNormal

# Create an instance of TruncatedLogNormal
prior1 = TruncatedLogNormal(0.1, 1.7, 0.001, 10)

# Plot the PDF for visualization
lower_bound = 0.001
upper_bound = 10.0
x_values = np.linspace(lower_bound, upper_bound, 1000)
pdf_values = prior1.pdf(torch.tensor(x_values))
plt.plot(x_values, pdf_values, label='PDF')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
