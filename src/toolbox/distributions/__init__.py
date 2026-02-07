from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import gamma as _gamma
from scipy.stats import laplace as _laplace
from scipy.stats import norm as _norm
from scipy.stats import uniform as _uniform
from scipy.stats._distn_infrastructure import rv_continuous

from .fit_moments import FitMomentsFactory, FitSpec
from .rectnorm import rectnorm
from .conditional import conditional


def _gamma_params(mean: float, std: float, _: Mapping[str, Any]) -> tuple[float, ...]:
    k = (mean / std) ** 2
    theta = mean / k
    return k, 0.0, theta  # a, loc, scale


def _gamma_freeze(params: tuple[float, ...]) -> rv_continuous:
    a, loc, scale = params
    return _gamma(a, loc=loc, scale=scale)


def _norm_params(mean: float, std: float, _: Mapping[str, Any]) -> tuple[float, ...]:
    return (mean, std)  # loc, scale


def _norm_freeze(params: tuple[float, ...]) -> rv_continuous:
    loc, scale = params
    return _norm(loc=loc, scale=scale)


def _laplace_params(mean: float, std: float, _: Mapping[str, Any]) -> tuple[float, ...]:
    # Laplace variance = 2*b^2 => std = sqrt(2)*b => b = std/sqrt(2)
    return (mean, float(std / np.sqrt(2.0)))  # loc, scale


def _laplace_freeze(params: tuple[float, ...]) -> rv_continuous:
    loc, scale = params
    return _laplace(loc=loc, scale=scale)


def _uniform_params(mean: float, std: float, _: Mapping[str, Any]) -> tuple[float, ...]:
    loc = mean - float(np.sqrt(3.0) * std)
    scale = float(2.0 * np.sqrt(3.0) * std)
    return (loc, scale)


def _uniform_freeze(params: tuple[float, ...]) -> rv_continuous:
    loc, scale = params
    return _uniform(loc=loc, scale=scale)


def _rectnorm_params(mean: float, std: float, _: Mapping[str, Any]) -> tuple[float, ...]:
    mu, sigma = rectnorm.params_from_mean_std(mean, std)
    return mu, sigma


def _rectnorm_freeze(params: tuple[float, ...]) -> rv_continuous:
    mu, sigma = params
    return rectnorm(mu, sigma)


registry = {
    "gamma": FitSpec(_gamma_params, _gamma_freeze),
    "norm": FitSpec(_norm_params, _norm_freeze),
    "laplace": FitSpec(_laplace_params, _laplace_freeze),
    "uniform": FitSpec(_uniform_params, _uniform_freeze),
    "rectnorm": FitSpec(_rectnorm_params, _rectnorm_freeze),
}

fit_moments = FitMomentsFactory(registry)

__all__ = [
    "fit_moments",
    "rectnorm",
    "conditional",
]
