import pytest

from modules.model_training import safe_division, validate_target_variable


# Tests
def test_validate_target_variable():
    validate_target_variable([1, 0, 1])
    with pytest.raises(ValueError):
        validate_target_variable([1, 1, 1])


def test_safe_division():
    assert safe_division(10, 2) == 5
    assert safe_division(10, 0) == 0
