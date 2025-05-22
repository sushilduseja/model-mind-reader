import pytest
import pandas as pd
from modules.data_ingestion import validate_csv_schema, parse_csv
from fastapi import HTTPException

# Fixtures
@pytest.fixture
def sample_csv():
    return """col1,col2,col3\n1,2,3\n4,5,6"""

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({"col1": [1, 4], "col2": [2, 5], "col3": [3, 6]})

# Tests
def test_validate_csv_schema(sample_dataframe):
    validate_csv_schema(sample_dataframe, ["col1", "col2"])
    with pytest.raises(HTTPException):
        validate_csv_schema(sample_dataframe, ["missing_col"])

def test_parse_csv(sample_csv):
    with open("test.csv", "w") as f:
        f.write(sample_csv)
    df = parse_csv("test.csv")
    assert not df.empty
    assert list(df.columns) == ["col1", "col2", "col3"]
