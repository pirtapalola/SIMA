"""

MODELS: Tools for data pre-processing and analysis
This code is used in the project "Simulation-based inference for marine remote sensing" by Palola et al.

TOOL NO.1 Calculate the minimum and maximum values in the simulated dataset.
TOOL NO.2 Wrap a sequence of PyTorch distributions into a joint PyTorch distribution.
TOOL N0.3 Create a truncated log-normal PyTorch distribution object.
TOOL NO.4 A function to fit a log-normal distribution to data.
TOOL NO.5 Check that the strings in a list have the same number of splits.
TOOL NO.6 Conduct min-max normalisation.

Last modified on 28 August 2024

"""

import numpy as np
import torch.nn as nn
import torch.optim as optim
import warnings
from typing import Dict, Optional, Sequence
import torch
from torch import Tensor
from torch.distributions import Distribution, constraints

"""

TOOL NO.1
Calculate the minimum and maximum values in the simulated dataset.

"""


def minimum_maximum(dataframe, column_names):
    print("Minimum and maximum values in the simulated dataset")
    for i in column_names:
        print(str(i) + " min: ")
        print(dataframe[i].min())
        print(str(i) + " max: ")
        print(dataframe[i].max())


"""

TOOL NO.2
Wrap a sequence of PyTorch distributions into a joint PyTorch distribution.

"""


class MultipleIndependent(Distribution):

    def __init__(
        self,
        dists: Sequence[Distribution],
        validate_args=None,
        arg_constraints: Dict[str, constraints.Constraint] = {},
    ):
        self._check_distributions(dists)
        if validate_args is not None:
            [d.set_default_validate_args(validate_args) for d in dists]

        self.dists = dists
        self.dims_per_dist = [d.sample().numel() for d in self.dists]

        self.ndims = int(torch.sum(torch.as_tensor(self.dims_per_dist)).item())
        self.custom_arg_constraints = arg_constraints
        self.validate_args = validate_args

        super().__init__(
            batch_shape=torch.Size([]),
            event_shape=torch.Size(
                [self.ndims]
            ),  # The sum of all ndims
            validate_args=validate_args,
        )

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.custom_arg_constraints

    def _check_distributions(self, dists):
        # Check that dists is a Sequence and that there is at least two distributions to combine.
        assert isinstance(
            dists, Sequence
        ), f"""The combination of independent priors must be of type Sequence, is
               {type(dists)}."""
        assert len(dists) > 1, "Provide at least two distributions to combine."
        # Check every element.
        [self._check_distribution(d) for d in dists]

    def _check_distribution(self, dist: Distribution):
        # Check the type and shape of a single input distribution.

        assert not isinstance(
            dist, (MultipleIndependent, Sequence)
        ), "It is not possible to nest the combined distributions."
        assert isinstance(
            dist, Distribution
        ), """Priors passed to MultipleIndependent must be PyTorch distributions. Make
            sure to process custom priors individually using process_prior before
            passing them to process_prior in a list."""
        # Check that batch shape is smaller or equal to 1.
        assert dist.batch_shape in (
            torch.Size([1]),
            torch.Size([0]),
            torch.Size([]),
        ), "The batch shape of every distribution should be smaller or equal to 1."

        assert (
            len(dist.batch_shape) > 0 or len(dist.event_shape) > 0
        ), """One of the distributions you passed is defined over a scalar only. 
        Distributions should have either event_shape > 0 or batch_shape > 0. For example,
            - instead of Uniform(0.0, 1.0) pass Uniform(torch.zeros(1), torch.ones(1))
        """

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        # Sample from every input distribution and concatenate samples.
        sample = torch.cat([d.sample(sample_shape) for d in self.dists], dim=-1)

        # Cover the case .sample() vs. .sample((n, )).
        if sample_shape == torch.Size():
            sample = sample.reshape(self.ndims)
        else:
            sample = sample.reshape(-1, self.ndims)

        return sample

    def log_prob(self, value) -> Tensor:
        value = self._prepare_value(value)

        # Evaluate value per distribution.
        num_samples = value.shape[0]
        log_probs = []
        dims_covered = 0
        for idx, d in enumerate(self.dists):
            ndims = int(self.dims_per_dist[idx])
            v = value[:, dims_covered: dims_covered + ndims]
            # Reshape to make sure that all returned log_probs are two-dimensional for concatenation.
            log_probs.append(d.log_prob(v).reshape(num_samples, 1))
            dims_covered += ndims

        # Sum across the last dimension to generate joint log probability over all inputted distributions.
        return torch.cat(log_probs, dim=1).sum(-1)

    def _prepare_value(self, value) -> Tensor:
        # This function raises an AssertionError if the value has over two dimensions.

        if value.ndim < 2:
            value = value.unsqueeze(0)

        assert (
            value.ndim == 2
        ), f"Value in log_prob should have ndim <= 2, instead it is {value.ndim}."

        batch_shape, num_value_dims = value.shape

        assert (
            num_value_dims == self.ndims
        ), f"The number of dimensions should match the dimensions of this joint distribution: {self.ndims}."

        return value

    @property
    def mean(self) -> Tensor:
        return torch.cat([d.mean for d in self.dists])

    @property
    def variance(self) -> Tensor:
        return torch.cat([d.variance for d in self.dists])

    @property
    def support(self):
        supports = []
        for d in self.dists:
            if isinstance(d.support, constraints.independent):
                supports.append(d.support.base_constraint)
            else:
                supports.append(d.support)
        # Wrap as 'independent' in order to have the correct shape.
        return constraints.independent(
            constraints.cat(supports, dim=1, lengths=self.dims_per_dist),
            reinterpreted_batch_ndims=1,
        )


def build_support(
    lower_bound: Optional[Tensor] = None, upper_bound: Optional[Tensor] = None
) -> constraints.Constraint:
    # lower_bound: lower bound of the prior
    # upper_bound: upper bound of the prior
    # A Pytorch constraint object is returned.

    if lower_bound is None and upper_bound is None:
        support = constraints.real
        warnings.warn(
            """No prior bounds were passed, consider defining lower_bound
            and upper_bound."""
        )

    # Only the lower bound is defined.
    elif upper_bound is None:
        num_dimensions = lower_bound.numel()  # type: ignore
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.greater_than(lower_bound),
                1,
            )
        else:
            support = constraints.greater_than(lower_bound)

    # Only the upper bound is defined.
    elif lower_bound is None:
        num_dimensions = upper_bound.numel()
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.less_than(upper_bound),
                1,
            )
        else:
            support = constraints.less_than(upper_bound)

    # Both the lower and upper bounds are defined.
    else:
        num_dimensions = lower_bound.numel()
        assert (
            num_dimensions == upper_bound.numel()
        ), "There should be an equal number of independent bounds."
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.interval(lower_bound, upper_bound),
                1,
            )
        else:
            support = constraints.interval(lower_bound, upper_bound)

    return support


"""

TOOL N0.3 
Create a truncated log-normal PyTorch distribution object.

"""


class TruncatedLogNormal(torch.distributions.Distribution):
    def __init__(self, loc, scale, upper_bound):
        super(TruncatedLogNormal, self).__init__()
        self.loc = loc
        self.scale = scale
        self.upper_bound = upper_bound
        self.lower_bound = 0.01  # Define a lower bound
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
            # Conduct truncation
            mask = (extra_samples >= self.lower_bound) & (extra_samples <= self.upper_bound)
            valid_samples = extra_samples[mask]
            generated_samples.append(valid_samples)
            total_samples += valid_samples.numel()
        # Concatenate the samples
        samples = torch.cat(generated_samples)[:sample_shape.numel()]
        # Print information to help debugging
        print(f"sample_shape: {sample_shape}, samples.size(): {samples.size()}, total_samples: {total_samples}")
        return samples

    def log_prob(self, value):
        # Calculate log probability
        log_prob_base = self.base_lognormal.log_prob(value)

        # Do not consider values outside the truncated range
        mask = (value >= self.lower_bound) & (value <= self.upper_bound)
        log_prob_truncated = torch.where(mask, log_prob_base, torch.tensor(float('-inf')))

        return log_prob_truncated

    def cdf(self, value):
        # Define the cumulative distribution function
        lower_bound_tensor = torch.tensor(self.lower_bound, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.float32)
        transformed_value = (torch.max(value, lower_bound_tensor) - self.loc) / self.scale
        transformed_value_tensor = torch.tensor(transformed_value, dtype=torch.float32)
        cdf_transformed = self.base_lognormal.cdf(transformed_value_tensor)

        return cdf_transformed

    def pdf(self, x):
        # Define the probability density function
        lower_bound_tensor = torch.tensor(self.lower_bound, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        log_pdf_base = self.base_lognormal.log_prob(torch.max(x, lower_bound_tensor))
        pdf_truncated = torch.exp(log_pdf_base - torch.log(self.cdf(self.upper_bound)))

        return pdf_truncated


"""

TOOL NO.4 
A function to fit a log-normal distribution to data using PyTorch.

"""


class LogNormalFitter(nn.Module):
    def __init__(self):
        super(LogNormalFitter, self).__init__()
        self.mu = nn.Parameter(torch.tensor(0.0))
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def forward(self, data):
        log_likelihood = torch.sum(
            -0.5 * ((torch.log(data) - self.mu) / self.sigma) ** 2 - torch.log(data * self.sigma) - 0.5 * np.log(
                2 * np.pi))
        return -log_likelihood


def fit_lognormal_torch(data):
    data = torch.tensor(data, dtype=torch.float32)

    model = LogNormalFitter()
    optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        return loss

    optimizer.step(closure)

    mu, sigma = model.mu, torch.exp(model.sigma)
    return mu.item(), sigma.item()


"""

TOOL NO.5 
Check that the strings in a list have the same number of splits.

"""


def find_strings_with_different_splits(list_of_strings, reference_string):
    reference_splits = len(reference_string.split("_"))
    different_splits_strings = []

    for s in list_of_strings:
        current_splits = len(s.split("_"))
        if current_splits != reference_splits:
            different_splits_strings.append(s)

    return different_splits_strings


"""

TOOL NO. 6
Conduct min-max normalisation.

"""


def min_max_normalisation(data_list):
    normalised = []
    list_min = data_list.min()
    list_max = data_list.max()
    for i in range(len(data_list)):
        normalised.append((data_list[i]-list_min)/(list_max-list_min))
    return normalised
