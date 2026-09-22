import pytest

from modules.explainability import validate_feature_alignment


# Tests
def test_validate_feature_alignment():
    good = ["feature1", "feature2"]
    bad = ["feature1", "feature3"]
    validate_feature_alignment(good, good)
    with pytest.raises(ValueError):
        validate_feature_alignment(good, bad)
