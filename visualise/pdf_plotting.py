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
upper_bound = 3
prior1 = TruncatedLogNormal(loc=0, scale=4, upper_bound=upper_bound)
prior2 = TruncatedLogNormal(loc=0, scale=5, upper_bound=upper_bound)
prior3 = TruncatedLogNormal(loc=0, scale=7, upper_bound=upper_bound)

# Plot the PDF for visualization
x_values = np.linspace(0.01, upper_bound, 1000)
pdf_values1 = prior1.pdf(torch.tensor(x_values))
pdf_values2 = prior2.pdf(torch.tensor(x_values))
pdf_values3 = prior3.pdf(torch.tensor(x_values))

plt.plot(x_values, pdf_values1, label='PDF 1')
plt.plot(x_values, pdf_values2, label='PDF 2')
plt.plot(x_values, pdf_values3, label='PDF 3')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

# Numerically integrate the PDF
area, _ = quad(lambda x: prior1.pdf(torch.tensor(x)), 0, upper_bound)
area_simps = simps(pdf_values1, x_values)  # Simpson's rule
print("Area under the PDF:", area)
print("Area under the PDF (Simpson's rule):", area_simps)
