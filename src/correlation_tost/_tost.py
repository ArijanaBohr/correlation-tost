from __future__ import annotations

import numpy as np
from scipy.stats import norm


def pearson_tost(
    observed: float,
    target: float,
    n: int,
    margin: float = 0.10,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Test equivalence between an observed and target Pearson correlation.

    Equivalence is assessed using two one-sided tests (TOST) after applying
    Fisher's z transformation to the observed correlation and equivalence
    bounds.

    Parameters
    ----------
    observed
        Observed Pearson correlation coefficient.
    target
        Target Pearson correlation coefficient.
    n
        Sample size used to estimate the observed correlation.
    margin
        Equivalence margin in Pearson correlation units. The equivalence
        interval is ``[target - margin, target + margin]``.
    alpha
        Significance level for each one-sided test.

    Returns
    -------
    dict
        Test p-value, one-sided p-values, test statistics, confidence
        interval, and equivalence decision.
    """
    if n <= 3:
        raise ValueError("n must be greater than 3.")

    if not -1 <= observed <= 1:
        raise ValueError("observed must be between -1 and 1.")

    if not -1 <= target <= 1:
        raise ValueError("target must be between -1 and 1.")

    if margin <= 0:
        raise ValueError("margin must be greater than 0.")

    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5.")

    eps = 1e-12

    observed = float(np.clip(observed, -1 + eps, 1 - eps))
    lower = float(np.clip(target - margin, -1 + eps, 1 - eps))
    upper = float(np.clip(target + margin, -1 + eps, 1 - eps))

    z_observed = np.arctanh(observed)
    z_lower = np.arctanh(lower)
    z_upper = np.arctanh(upper)

    se = 1.0 / np.sqrt(n - 3)

    # Lower one-sided test:
    # H0: rho <= lower
    # H1: rho > lower
    z_stat_lower = (z_observed - z_lower) / se
    p_lower = norm.sf(z_stat_lower)

    # Upper one-sided test:
    # H0: rho >= upper
    # H1: rho < upper
    z_stat_upper = (z_observed - z_upper) / se
    p_upper = norm.cdf(z_stat_upper)

    # Both one-sided tests must reject for equivalence.
    p_value = max(p_lower, p_upper)

    # 100 * (1 - 2 * alpha)% confidence interval.
    z_crit = norm.ppf(1 - alpha)
    ci_lower = np.tanh(z_observed - z_crit * se)
    ci_upper = np.tanh(z_observed + z_crit * se)

    return {
        "p_value": float(p_value),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "z_stat_lower": float(z_stat_lower),
        "z_stat_upper": float(z_stat_upper),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "equivalent": bool(p_value < alpha),
    }
