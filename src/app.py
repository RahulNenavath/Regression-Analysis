import os
from pathlib import Path
import streamlit as st
import pandas as pd
from regression import LinearRegressionNumpy

project_path = Path(os.getcwd())
lr = LinearRegressionNumpy()

def main():
    st.title("Regression Analysis from Scratch")

    # Step 1: Select file format
    file_format = st.selectbox("Select file format", options=["CSV", "Parquet"])

    # Step 2: Show file uploader for supported formats
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'parquet'])

    # Check if the uploaded file and format match
    if uploaded_file:
        try:
            # Step 3: Validate and read columns based on selected file format
            if file_format == "CSV" and uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif file_format == "Parquet" and uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("File format mismatch. Please upload a file in the selected format.")
                return

            # Extract column names and show dropdown for response variable selection
            columns = df.columns.to_list()

            # Step 4: Select response variable from the columns
            response_variable = st.selectbox("Select the response variable column", options=columns)

            # Step 5: Rename Response Variable
            df = df.rename(columns={response_variable : 'y'})

            features, target = df.drop('y', axis=1), df['y']

            # Analyze button to upload the file to GCS
            if st.button("Analyze"):
                lr.fit(features, target)
                print(lr.summary())
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

if __name__ == "__main__":
    main()