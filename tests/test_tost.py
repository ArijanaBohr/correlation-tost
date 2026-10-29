import pytest

from correlation_tost import pearson_tost


def test_matches_toster_reference_case():
    """Match a reference result from TOSTER::corsum_test."""
    result = pearson_tost(
        observed=0.30,
        target=0.30,
        n=10_000,
        margin=0.10,
        alpha=0.05,
    )

    assert result["p_value"] == pytest.approx(
        6.513681924396496e-27,
        rel=1e-12,
    )
    assert result["ci_lower"] == pytest.approx(
        0.2849567,
        abs=1e-7,
    )
    assert result["ci_upper"] == pytest.approx(
        0.3148956,
        abs=1e-7,
    )
    assert result["equivalent"] is True


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 3}, "n must be greater than 3"),
        ({"observed": 1.1}, "observed must be between -1 and 1"),
        ({"target": -1.1}, "target must be between -1 and 1"),
        ({"margin": 0.0}, "margin must be greater than 0"),
        ({"alpha": 0.0}, "alpha must be between 0 and 0.5"),
        ({"alpha": 0.5}, "alpha must be between 0 and 0.5"),
    ],
)
def test_invalid_inputs(kwargs, message):
    params = {
        "observed": 0.30,
        "target": 0.30,
        "n": 100,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        pearson_tost(**params)
