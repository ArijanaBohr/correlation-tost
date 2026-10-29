# correlation-tost

A lightweight Python implementation of the two one-sided tests (TOST) procedure for equivalence testing of Pearson correlations.

The test uses Fisher's z transformation and has been numerically validated against R's `TOSTER::corsum_test`.

## Installation

Install directly from GitHub with `pip`:

```bash
pip install git+https://github.com/ArijanaBohr/correlation-tost.git
```

**Requirements:** Python ≥ 3.9, NumPy, SciPy

## Quick start


```python
from correlation_tost import pearson_tost

result = pearson_tost(
    observed=0.30,  # observed Pearson correlation
    target=0.30,    # target Pearson correlation
    n=10_000,       # sample size
    margin=0.10,    # equivalence margin
    alpha=0.05,     # significance level
)

print(result["equivalent"])  # True / False
print(result["p_value"])     # overall TOST p-value
```

The equivalence bounds are defined in Pearson correlation units as
`[target - margin, target + margin]`. Fisher's z transformation is applied
internally when performing the test.

## Output

`pearson_tost` returns a dictionary with the following keys:

| Key            | Description                                                   |
|----------------|---------------------------------------------------------------|
| `p_value`      | Overall TOST p-value                                          |
| `p_lower`      | p-value for the lower-bound test                              |
| `p_upper`      | p-value for the upper-bound test                              |
| `z_stat_lower` | Test statistic for the lower-bound test                       |
| `z_stat_upper` | Test statistic for the upper-bound test                       |
| `ci_lower`     | Lower confidence interval bound                               |
| `ci_upper`     | Upper confidence interval bound                               |
| `equivalent`   | Whether equivalence is concluded at the specified `alpha`     |

## Validation

The implementation was independently validated against
`TOSTER::corsum_test(method = "pearson", alternative = "equivalence")`
across:

- positive, negative, and zero correlations
- multiple sample sizes
- multiple equivalence margins
- a range of correlation strengths

The package itself does **not** depend on R, rpy2, or TOSTER.
