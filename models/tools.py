"""

Tools for data pre-processing and analysis

TOOL NO.1 Calculate the minimum and maximum values in the simulated dataset.
TOOL NO.2 Create a pandas dataframe containing the input parameters (each row corresponds to a single simulation run).
TOOL NO.3 Wrap a sequence of PyTorch distributions into a joint PyTorch distribution.
TOOL N0.4 Create a truncated log-normal PyTorch distribution object.
TOOL NO.5 A function to fit a log-normal distribution to data.
TOOL NO.6 Check that the strings in a list have the same number of splits.
TOOL NO.7 Get the values of the parameters of each simulation run from the filenames.

Last modified on 18 December 2023 by Pirta Palola.

"""

import numpy as np
import torch.nn as nn
import torch.optim as optim
import warnings
from typing import Dict, Optional, Sequence
import torch
from torch import Tensor
from torch.distributions import Distribution, constraints
import pandas as pd

"""

TOOL NO.1
Calculate the minimum and maximum values in the simulated dataset

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
Create a pandas dataframe containing the input parameters (each row corresponds to a single simulation run).

"""


def create_input_dataframe(list_of_strings):
    split_df = pd.DataFrame(columns=["data", "water", "phy1", "cdom1", "spm1", "wind1", "depth1"])
    phy_list = []
    cdom_list = []
    spm_list = []
    wind_list = []
    depth_list = []
    depth_list0 = []

    for i in list_of_strings:
        split_string = i.split("_")  # Split the string at the locations marked by underscores
        split_df.loc[len(split_df)] = split_string  # Add the split string as a row in the dataframe

    for n in split_df["phy1"]:  # Create a list where the decimal dots are added
        phy_list.append(float(n[:1] + '.' + n[1:]))
    split_df["phy"] = phy_list  # Create a new column that contains the values with decimal dots

    for n in split_df["cdom1"]:  # Create a list where the decimal dots are added
        cdom_list.append(float(n[:1] + '.' + n[1:]))
    split_df["cdom"] = cdom_list  # Create a new column that contains the values with decimal dots

    for n in split_df["spm1"]:  # Create a list where the decimal dots are added
        spm_list.append(float(n[:1] + '.' + n[1:]))
    split_df["spm"] = spm_list  # Create a new column that contains the values with decimal dots

    for n in split_df["wind1"]:  # Create a list where the decimal dots are added
        wind_list.append(float(n[:1] + '.' + n[1:]))
    split_df["wind"] = wind_list  # Create a new column that contains the values with decimal dots

    for n in split_df["depth1"]:
        sep = '.'
        depth_list0.append(n.split(sep, 1)[0])  # Remove ".txt" from the string based on the separator "."

    #for x in depth_list0:  # Create a list where the decimal dots are added
     #   depth_list.append(float(x[:1] + '.' + x[1:]))
    split_df["depth"] = depth_list0  # Create a new column that contains the values with decimal dots

    # Drop the columns that do not contain the values to be inferred
    split_df = split_df.drop(columns=["data", "water", "phy1", "cdom1", "spm1", "wind1", "depth1", "depth"])
    return split_df


"""

TOOL NO.3
Wrap a sequence of PyTorch distributions into a joint PyTorch distribution.

    Every element of the sequence is treated as independent from the other elements.
    Single elements can be multivariate with dependent dimensions, e.g.,:
        - [
            Gamma(torch.zeros(1), torch.ones(1)),
            Beta(torch.zeros(1), torch.ones(1)),
            MVG(torch.ones(2), torch.tensor([[1, .1], [.1, 1.]]))
        ]
        - [
            Uniform(torch.zeros(1), torch.ones(1)),
            Uniform(torch.ones(1), 2.0 * torch.ones(1))]

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
        # numel() instead of event_shape because for all dists both is possible,
        # event_shape=[1] or batch_shape=[1]
        self.dims_per_dist = [d.sample().numel() for d in self.dists]

        self.ndims = int(torch.sum(torch.as_tensor(self.dims_per_dist)).item())
        self.custom_arg_constraints = arg_constraints
        self.validate_args = validate_args

        super().__init__(
            batch_shape=torch.Size([]),  # batch size was ensured to be <= 1 above.
            event_shape=torch.Size(
                [self.ndims]
            ),  # Event shape is the sum of all ndims.
            validate_args=validate_args,
        )

    @property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        return self.custom_arg_constraints

    def _check_distributions(self, dists):
        """Check if dists is Sequence and longer 1 and check every member."""
        assert isinstance(
            dists, Sequence
        ), f"""The combination of independent priors must be of type Sequence, is
               {type(dists)}."""
        assert len(dists) > 1, "Provide at least 2 distributions to combine."
        # Check every element of the sequence.
        [self._check_distribution(d) for d in dists]

    def _check_distribution(self, dist: Distribution):
        """Check type and shape of a single input distribution."""

        assert not isinstance(
            dist, (MultipleIndependent, Sequence)
        ), "Nesting of combined distributions is not possible."
        assert isinstance(
            dist, Distribution
        ), """priors passed to MultipleIndependent must be PyTorch distributions. Make
            sure to process custom priors individually using process_prior before
            passing them in a list to process_prior."""
        # Make sure batch shape is smaller or equal to 1.
        assert dist.batch_shape in (
            torch.Size([1]),
            torch.Size([0]),
            torch.Size([]),
        ), "The batch shape of every distribution must be smaller or equal to 1."

        assert (
            len(dist.batch_shape) > 0 or len(dist.event_shape) > 0
        ), """One of the distributions you passed is defined over a scalar only. Make
        sure pass distributions with one of event_shape or batch_shape > 0: For example
            - instead of Uniform(0.0, 1.0) pass Uniform(torch.zeros(1), torch.ones(1))
            - instead of Beta(1.0, 2.0) pass Beta(tensor([1.0]), tensor([2.0])).
        """

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        # Sample from every sub distribution and concatenate samples.
        sample = torch.cat([d.sample(sample_shape) for d in self.dists], dim=-1)

        # This reshape is needed to cover the case .sample() vs. .sample((n, )).
        if sample_shape == torch.Size():
            sample = sample.reshape(self.ndims)
        else:
            sample = sample.reshape(-1, self.ndims)

        return sample

    def log_prob(self, value) -> Tensor:
        value = self._prepare_value(value)

        # Evaluate value per distribution, taking into account that individual
        # distributions can be multivariate.
        num_samples = value.shape[0]
        log_probs = []
        dims_covered = 0
        for idx, d in enumerate(self.dists):
            ndims = int(self.dims_per_dist[idx])
            v = value[:, dims_covered : dims_covered + ndims]
            # Reshape here to ensure all returned log_probs are 2D for concatenation.
            log_probs.append(d.log_prob(v).reshape(num_samples, 1))
            dims_covered += ndims

        # Sum accross last dimension to get joint log prob over all distributions.
        return torch.cat(log_probs, dim=1).sum(-1)

    def _prepare_value(self, value) -> Tensor:
        """Return input value with fixed shape.

        Raises:
            AssertionError: if value has more than 2 dimensions or invalid size in
                2nd dimension.
        """

        if value.ndim < 2:
            value = value.unsqueeze(0)

        assert (
            value.ndim == 2
        ), f"value in log_prob must have ndim <= 2, it is {value.ndim}."

        batch_shape, num_value_dims = value.shape

        assert (
            num_value_dims == self.ndims
        ), f"Number of dimensions must match dimensions of this joint: {self.ndims}."

        return value

    @property
    def mean(self) -> Tensor:
        return torch.cat([d.mean for d in self.dists])

    @property
    def variance(self) -> Tensor:
        return torch.cat([d.variance for d in self.dists])

    @property
    def support(self):
        # First, we remove all `independent` constraints. This applies to e.g.
        # `MultivariateNormal`. An `independent` constraint returns a 1D `[True]`
        # when `.support.check(sample)` is called, whereas distributions that are
        # not `independent` (e.g. `Gamma`), return a 2D `[[True]]`. When such
        # constraints would be combined with the `constraint.cat(..., dim=1)`, it
        # fails because the `independent` constraint returned only a 1D `[True]`.
        supports = []
        for d in self.dists:
            if isinstance(d.support, constraints.independent):
                supports.append(d.support.base_constraint)
            else:
                supports.append(d.support)
        # Wrap as `independent` in order to have the correct shape of the
        # `log_abs_det`, i.e. summed over the parameter dimensions.
        return constraints.independent(
            constraints.cat(supports, dim=1, lengths=self.dims_per_dist),
            reinterpreted_batch_ndims=1,
        )


def build_support(
    lower_bound: Optional[Tensor] = None, upper_bound: Optional[Tensor] = None
) -> constraints.Constraint:
    """Return support for prior distribution, depending on available bounds.

    Args:
        lower_bound: lower bound of the prior support, can be None
        upper_bound: upper bound of the prior support, can be None

    Returns:
        support: Pytorch constraint object.
    """

    # Support is real if no bounds are passed.
    if lower_bound is None and upper_bound is None:
        support = constraints.real
        warnings.warn(
            """No prior bounds were passed, consider passing lower_bound
            and / or upper_bound if your prior has bounded support."""
        )
    # Only lower bound is specified.
    elif upper_bound is None:
        num_dimensions = lower_bound.numel()  # type: ignore
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.greater_than(lower_bound),
                1,
            )
        else:
            support = constraints.greater_than(lower_bound)
    # Only upper bound is specified.
    elif lower_bound is None:
        num_dimensions = upper_bound.numel()
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.less_than(upper_bound),
                1,
            )
        else:
            support = constraints.less_than(upper_bound)

    # Both are specified.
    else:
        num_dimensions = lower_bound.numel()
        assert (
            num_dimensions == upper_bound.numel()
        ), "There must be an equal number of independent bounds."
        if num_dimensions > 1:
            support = constraints._IndependentConstraint(
                constraints.interval(lower_bound, upper_bound),
                1,
            )
        else:
            support = constraints.interval(lower_bound, upper_bound)

    return support


"""TOOL N0.4 Create a truncated log-normal PyTorch distribution object."""


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

        while total_samples < torch.Size(sample_shape).numel():
            remaining_samples = torch.Size(sample_shape).numel() - total_samples
            extra_samples = self.base_lognormal.sample(torch.Size([remaining_samples]))

            # Apply truncation using vectorized operations
            mask = (extra_samples >= self.lower_bound) & (extra_samples <= self.upper_bound)
            valid_samples = extra_samples[mask]

            generated_samples.append(valid_samples)
            total_samples += valid_samples.numel()

        # Concatenate the generated samples
        samples = torch.cat(generated_samples)[:torch.Size(sample_shape).numel()]

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
        cdf_transformed = self.base_lognormal.cdf(transformed_value_tensor)

        return cdf_transformed

    def pdf(self, x):
        # Probability density function
        pdf_base = torch.exp(self.base_lognormal.log_prob(x))
        pdf_truncated = pdf_base / (self.cdf(self.upper_bound) - self.cdf(self.lower_bound))
        return pdf_truncated.numpy()


"""
class TruncatedLogNormal(torch.distributions.Distribution):
    def __init__(self, loc, scale, lower_bound, upper_bound):
        self.loc = loc
        self.scale = scale
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.base_lognormal = torch.distributions.LogNormal(loc, scale)

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
        return self.base_lognormal.cdf(value)"""


"""TOOL NO.5 A function to fit a log-normal distribution to data using PyTorch."""


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


"""TOOL NO.6 Check that the strings in a list have the same number of splits."""


def find_strings_with_different_splits(list_of_strings, reference_string):
    reference_splits = len(reference_string.split("_"))  # Split at the locations marked by an underscore.
    different_splits_strings = []

    for s in list_of_strings:
        current_splits = len(s.split("_"))
        if current_splits != reference_splits:
            different_splits_strings.append(s)

    return different_splits_strings


"""TOOL NO.7 Get the values of the parameters of each simulation run from the filenames."""


def extract_values_from_filename(filename):
    # Remove ".txt" from the filename
    filename = filename.replace(".txt", "")

    # Assuming the format "Mcoralbrown_00_00_021_461_672_100"
    parts = filename.split('_')

    # Extract parameter values from the filename
    try:
        water = int(parts[1]) / 100.0
        phy = int(parts[2]) / 100.0
        cdom = int(parts[3]) / 100.0
        spm = int(parts[4]) / 100.0
        wind = int(parts[5]) / 100.0
        depth = int(parts[6]) / 10.0
    except ValueError:
        return filename  # Return filename if there's an error

    return water, phy, cdom, spm, wind, depth
