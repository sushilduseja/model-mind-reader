import pandas as pd
import pytest

from modules.fairness_metrics import (
    compute_rate_safe,
    validate_protected_attribute,
)


# Fixtures
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"protected": ["A", "B"], "value": [1, 2]})


# Tests
def test_validate_protected_attribute(sample_dataframe):
    validate_protected_attribute(sample_dataframe, "protected")
    with pytest.raises(ValueError):
        validate_protected_attribute(sample_dataframe, "missing")


def test_compute_rate_safe():
    assert compute_rate_safe(10, 2) == 5
    assert compute_rate_safe(10, 0) == 0
