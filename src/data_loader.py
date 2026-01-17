"""
Data Loader Module

Responsible for loading and validating historical stock data from CSV files.

This module provides functionality to:
- Load CSV files from the data/ directory
- Validate data structure and content
- Return cleaned DataFrames ready for analysis
"""

import pandas as pd
from pathlib import Path


def load_stock_data(company: str, data_dir: str = "data") -> pd.DataFrame:
    """
    Load historical stock data for a given company from CSV files.
    
    This function:
    - Loads CSV data from data/{company}.csv
    - Parses Date column as datetime
    - Ensures Close price column exists (with 'Adj Close' fallback)
    - Sorts by Date ascending
    - Filters to last 1 year of data
    
    Args:
        company (str): Company ticker symbol (e.g., 'TSLA', 'AAPL', 'AMZN')
        data_dir (str): Directory path containing CSV files (default: 'data')
    
    Returns:
        pd.DataFrame: DataFrame with columns including Date, Close, and other price data
    
    Raises:
        FileNotFoundError: If the CSV file for the company does not exist
        ValueError: If required columns are missing
    """
    # Build path to CSV file
    data_path = Path(data_dir) / f"{company}.csv"
    
    # Check if file exists - raise clear error if not
    if not data_path.exists():
        raise FileNotFoundError(
            f"CSV file not found for company: {company}\n"
            f"Expected file at: {data_path}\n"
            f"Please ensure the file exists in the data directory."
        )
    
    # Load CSV file into DataFrame
    df = pd.read_csv(data_path)
    
    # Parse Date column as datetime
    # Handle different date formats that might exist in CSV files
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    else:
        raise ValueError("CSV file must contain a 'Date' column")
    
    # Ensure Close price column exists
    # Try 'Close' first, then fall back to 'Adj Close' if needed
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            # Use Adj Close as Close if Close doesn't exist
            df['Close'] = df['Adj Close']
        elif ' Close/Last' in df.columns:
            # Handle AAPL format with space and slash
            df['Close'] = df[' Close/Last'].str.replace('$', '').str.replace(',', '').astype(float)
        else:
            raise ValueError(
                "CSV file must contain either 'Close' or 'Adj Close' column"
            )
    
    # Clean Close column - remove any string formatting like $ signs
    if df['Close'].dtype == 'object':
        df['Close'] = df['Close'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Sort by Date ascending (oldest first)
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # Filter to last 1 year of data
    # Get the most recent date in the dataset
    latest_date = df['Date'].max()
    # Calculate date 1 year ago from the latest date
    one_year_ago = latest_date - pd.DateOffset(years=1)
    # Filter DataFrame to only include last year
    df = df[df['Date'] >= one_year_ago].reset_index(drop=True)
    
    return df


def get_available_tickers(data_dir: str = "data") -> list:
    """
    Get list of available stock tickers from CSV files in data directory.
    
    Args:
        data_dir (str): Directory path to search for CSV files
    
    Returns:
        list: List of available ticker symbols
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return []
    
    # Find all CSV files and extract ticker symbols (filename without extension)
    tickers = [f.stem for f in data_path.glob("*.csv")]
    
    return sorted(tickers)


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded DataFrame has required columns and data types.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
    
    Returns:
        bool: True if valid, raises exception otherwise
    """
    required_columns = ['Date', 'Close']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check that Close column has numeric values
    if not pd.api.types.is_numeric_dtype(df['Close']):
        raise ValueError("Close column must contain numeric values")
    
    return True
