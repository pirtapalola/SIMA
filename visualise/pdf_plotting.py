"""

Plot pdf for a Truncated LogNormal distribution.

"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.tools import TruncatedLogNormal
from scipy.integrate import quad, simps

# Create an instance of TruncatedLogNormal
prior1 = TruncatedLogNormal(loc=0.1, scale=1.7, upper_bound=10)

# Plot the PDF for visualization
upper_bound = 10
x_values = np.linspace(0.01, upper_bound, 1000)
pdf_values = prior1.pdf(torch.tensor(x_values))

# Ensure the values are within the support (greater than or equal to zero)
plt.plot(x_values, pdf_values, label='PDF')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Numerically integrate the PDF with simps (Simpson's rule)
area_simps = simps(pdf_values, x_values)

print("Area under the PDF (Simpson's rule):", area_simps)
