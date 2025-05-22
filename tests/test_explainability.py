import pytest
from modules.explainability import validate_feature_alignment

# Tests
def test_validate_feature_alignment():
    validate_feature_alignment(["feature1", "feature2"], ["feature1", "feature2"])
    with pytest.raises(ValueError):
        validate_feature_alignment(["feature1", "feature2"], ["feature1", "feature3"])
