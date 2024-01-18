"""

Plot pdf for a Truncated LogNormal distribution.

"""

# Import libraries
import numpy as np
import matplotlib as plt
from scipy.integrate import quad
import torch


class TruncatedLogNormal(torch.distributions.Distribution):
    def __init__(self, loc, scale, lower_bound, upper_bound):
        self.loc = loc
        self.scale = scale
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.base_lognormal = torch.distributions.LogNormal(loc, scale)

    @property
    def batch_shape(self):
        return self.base_lognormal.batch_shape

    @property
    def event_shape(self):
        return self.base_lognormal.event_shape

    def sample(self, sample_shape=torch.Size()):
        generated_samples = []
        total_samples = 0

        while total_samples < sample_shape.numel():
            remaining_samples = sample_shape.numel() - total_samples
            extra_samples = self.base_lognormal.sample(torch.Size([remaining_samples]))

            # Apply truncation using vectorized operations
            mask = (extra_samples >= self.lower_bound) & (extra_samples <= self.upper_bound)
            valid_samples = extra_samples[mask]

            generated_samples.append(valid_samples)
            total_samples += valid_samples.numel()

        # Concatenate the generated samples
        samples = torch.cat(generated_samples)[:sample_shape.numel()]

        # Debugging information
        print(f"sample_shape: {sample_shape}, samples.size(): {samples.size()}, total_samples: {total_samples}")

        return samples

    def log_prob(self, value):
        # Calculate log probability for a given value using vectorized operations
        log_prob_base = self.base_lognormal.log_prob(value)
        log_prob_truncated = log_prob_base - torch.log(self.cdf(self.upper_bound) - self.cdf(self.lower_bound))

        return log_prob_truncated

    def cdf(self, value):
        # Cumulative distribution function
        transformed_value = (value - self.loc) / self.scale
        transformed_value_tensor = torch.tensor(transformed_value, dtype=torch.float32)

        # Check for problematic values explicitly
        if torch.isnan(transformed_value_tensor).any():
            nan_values = transformed_value_tensor[torch.isnan(transformed_value_tensor)].detach().cpu().numpy()
            print(f"Problematic values: {nan_values}")
            raise ValueError("Invalid value in the transformed tensor.")

        cdf_transformed = self.base_lognormal.cdf(transformed_value_tensor)

        return cdf_transformed

    def pdf(self, x):
        # Probability density function
        log_prob_base = self.base_lognormal.log_prob(x)
        pdf_base = torch.exp(log_prob_base)

        cdf_upper = self.cdf(self.upper_bound)
        cdf_lower = self.cdf(self.lower_bound)

        # Print problematic values in the final PDF
        nan_indices_pdf = torch.isnan(cdf_upper) | torch.isnan(cdf_lower)
        if nan_indices_pdf.any():
            nan_values_pdf = x[nan_indices_pdf].detach().cpu().numpy()
            print(f"Problematic values in cdf: {nan_values_pdf}")
            raise ValueError("Invalid value in the cdf.")

        pdf_truncated = pdf_base / (cdf_upper - cdf_lower)

        # Print problematic values in the final PDF
        nan_indices_pdf = torch.isnan(pdf_truncated)
        if nan_indices_pdf.any():
            nan_values_pdf = x[nan_indices_pdf].detach().cpu().numpy()
            print(f"Problematic values in final PDF: {nan_values_pdf}")
            raise ValueError("Invalid value in the final PDF.")

        return pdf_truncated.numpy()


# Create an instance of TruncatedLogNormal
prior1 = TruncatedLogNormal(0.1, 1.7, 0.001, 10)

# Define the integration range
lower_bound = 0.001
upper_bound = 10


# Define the function to integrate
def pdf_function(x):
    try:
        result = prior1.pdf(torch.tensor(x))
    except ValueError as e:
        print(f"Error in pdf_function: {e}")
        result = 0.0  # Return 0.0 if there is an error

    return result


x_values_to_check = np.linspace(lower_bound, upper_bound, 1000)
pdf_values_to_check = pdf_function(x_values_to_check)
print("pdf_function values to check:", pdf_values_to_check)

# Integrate the PDF over the range
result, _ = quad(pdf_function, lower_bound, upper_bound)

print(f"Integral result: {result}")

# Plot the PDF for visualization
x_values = np.linspace(lower_bound, upper_bound, 1000)
pdf_values = prior1.pdf(torch.tensor(x_values))
plt.plot(x_values, pdf_values, label='PDF')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
