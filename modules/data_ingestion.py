from typing import List

import pandas as pd
import streamlit as st
from fastapi import HTTPException


def validate_csv_schema(data: pd.DataFrame, required: List[str]) -> None:
    """Validate that the required columns exist in the DataFrame."""
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required columns",
                "missing_columns": missing,
            },
        )


def parse_csv(file: str) -> pd.DataFrame:
    """Parse a CSV file into a DataFrame."""
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "Failed to parse CSV file", "details": str(e)},
        )


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Perform preprocessing on the DataFrame (e.g., handle missing values)."""
    # Example: Fill missing values with 0
    return data.fillna(0)


def upload_and_validate_data():
    """Handles CSV file upload and schema validation."""
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Data Preview:", data.head())
            return data
        except Exception as e:
            st.error(f"Error reading the file: {e}")

    return None
