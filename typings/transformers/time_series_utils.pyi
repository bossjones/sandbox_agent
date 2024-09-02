"""
This type stub file was generated by pyright.
"""

import torch
from typing import Callable, Dict, Optional, Tuple
from torch import nn
from torch.distributions import Distribution, TransformedDistribution

"""
Time series distributional output classes and utilities.
"""
class AffineTransformed(TransformedDistribution):
    def __init__(self, base_distribution: Distribution, loc=..., scale=..., event_dim=...) -> None:
        ...

    @property
    def mean(self):
        """
        Returns the mean of the distribution.
        """
        ...

    @property
    def variance(self):
        """
        Returns the variance of the distribution.
        """
        ...

    @property
    def stddev(self):
        """
        Returns the standard deviation of the distribution.
        """
        ...



class ParameterProjection(nn.Module):
    def __init__(self, in_features: int, args_dim: Dict[str, int], domain_map: Callable[..., Tuple[torch.Tensor]], **kwargs) -> None:
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        ...



class LambdaLayer(nn.Module):
    def __init__(self, function) -> None:
        ...

    def forward(self, x, *args):
        ...



class DistributionOutput:
    distribution_class: type
    in_features: int
    args_dim: Dict[str, int]
    def __init__(self, dim: int = ...) -> None:
        ...

    def distribution(self, distr_args, loc: Optional[torch.Tensor] = ..., scale: Optional[torch.Tensor] = ...) -> Distribution:
        ...

    @property
    def event_shape(self) -> Tuple:
        r"""
        Shape of each individual event contemplated by the distributions that this object constructs.
        """
        ...

    @property
    def event_dim(self) -> int:
        r"""
        Number of event dimensions, i.e., length of the `event_shape` tuple, of the distributions that this object
        constructs.
        """
        ...

    @property
    def value_in_support(self) -> float:
        r"""
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
        ...

    def get_parameter_projection(self, in_features: int) -> nn.Module:
        r"""
        Return the parameter projection layer that maps the input to the appropriate parameters of the distribution.
        """
        ...

    def domain_map(self, *args: torch.Tensor):
        r"""
        Converts arguments to the right shape and domain. The domain depends on the type of distribution, while the
        correct shape is obtained by reshaping the trailing axis in such a way that the returned tensors define a
        distribution of the right event_shape.
        """
        ...

    @staticmethod
    def squareplus(x: torch.Tensor) -> torch.Tensor:
        r"""
        Helper to map inputs to the positive orthant by applying the square-plus operation. Reference:
        https://twitter.com/jon_barron/status/1387167648669048833
        """
        ...



class StudentTOutput(DistributionOutput):
    """
    Student-T distribution output class.
    """
    args_dim: Dict[str, int] = ...
    distribution_class: type = ...
    @classmethod
    def domain_map(cls, df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor): # -> tuple[Tensor, Tensor, Tensor]:
        ...



class NormalOutput(DistributionOutput):
    """
    Normal distribution output class.
    """
    args_dim: Dict[str, int] = ...
    distribution_class: type = ...
    @classmethod
    def domain_map(cls, loc: torch.Tensor, scale: torch.Tensor): # -> tuple[Tensor, Tensor]:
        ...



class NegativeBinomialOutput(DistributionOutput):
    """
    Negative Binomial distribution output class.
    """
    args_dim: Dict[str, int] = ...
    distribution_class: type = ...
    @classmethod
    def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor): # -> tuple[Tensor, Tensor]:
        ...

    def distribution(self, distr_args, loc: Optional[torch.Tensor] = ..., scale: Optional[torch.Tensor] = ...) -> Distribution:
        ...

