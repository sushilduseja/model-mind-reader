import streamlit as st
import pandas as pd

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
