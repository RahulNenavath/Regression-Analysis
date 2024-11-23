
# Regression Analysis from Scratch

This project demonstrates a regression analysis pipeline using a custom linear regression model implemented from scratch in Numpy & Python. The application provides a Streamlit-based interface to analyze datasets and perform Ordinary Least Squares (OLS) regression.

## Features

- **Custom Linear Regression Implementation**: Implements OLS regression with detailed statistical analysis, including confidence intervals, t-values, and F-statistics.
- **Streamlit Interface**: User-friendly web application to upload and analyze datasets in CSV or Parquet formats.
- **Interactive Data Analysis**: Allows users to select the response variable and automatically analyzes features.

## Requirements

To run this project, install the following dependencies:

```bash
pip install numpy pandas scipy streamlit
```

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RahulNenavath/Regression-Analysis.git
   cd regression-analysis-from-scratch
   ```

2. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

3. **Upload Dataset**:
   - Choose the file format (CSV or Parquet).
   - Upload the dataset.
   - Select the response variable column.
   - Click on the "Analyze" button to view the regression summary.