"""
Volatility Calculation Module

Responsible for computing volatility metrics using rolling statistics and Z-scores.

This module provides functionality to:
- Calculate rolling volatility (standard deviation)
- Compute Z-scores for anomaly detection
- Identify volatility spikes and events
"""

import pandas as pd
import numpy as np


def calculate_volatility(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Calculate volatility metrics using rolling statistics and Z-scores.
    
    This function computes:
    1. 7-day rolling mean of Close prices
    2. 7-day rolling standard deviation (volatility measure)
    3. Z-score: standardized deviation from the rolling mean
    
    Z-score interpretation:
    - Z-score near 0: Price is close to recent average
    - Z-score > 2: Price is significantly higher than usual (high volatility)
    - Z-score < -2: Price is significantly lower than usual (high volatility)
    
    Args:
        df (pd.DataFrame): DataFrame with at least 'Close' column
        window (int): Rolling window size in days (default: 7)
    
    Returns:
        pd.DataFrame: Original DataFrame with added volatility columns:
                      - rolling_mean: 7-day rolling mean of Close price
                      - rolling_std: 7-day rolling standard deviation
                      - z_score: (Close - rolling_mean) / rolling_std
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Step 1: Calculate 7-day rolling mean of Close prices
    # This shows the average price over the last 7 days
    # min_periods=1 means we calculate even if we have fewer than 7 days
    df['rolling_mean'] = df['Close'].rolling(window=window, min_periods=1).mean()
    
    # Step 2: Calculate 7-day rolling standard deviation
    # This measures how much prices vary from the mean (volatility)
    # Higher std = more volatile/unstable prices
    df['rolling_std'] = df['Close'].rolling(window=window, min_periods=1).std()
    
    # Step 3: Calculate Z-score for anomaly detection
    # Z-score = (actual_price - average_price) / standard_deviation
    # This standardizes the deviation from the mean
    
    # Handle division by zero safely:
    # If rolling_std is 0 (no variation in prices), we can't calculate Z-score
    # Replace 0 with NaN to avoid division errors
    df['rolling_std_safe'] = df['rolling_std'].replace(0, np.nan)
    
    # Calculate Z-score
    df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std_safe']
    
    # Drop the temporary safe column
    df = df.drop(columns=['rolling_std_safe'])
    
    # Fill any infinite values with NaN for safety
    df['z_score'] = df['z_score'].replace([np.inf, -np.inf], np.nan)
    
    return df


def calculate_rolling_volatility(
    prices: pd.Series,
    window: int = 20,
    min_periods: int = 1
) -> pd.Series:
    """
    Calculate rolling volatility (standard deviation) of price returns.
    
    Args:
        prices (pd.Series): Series of price data
        window (int): Rolling window size (default: 20 days)
        min_periods (int): Minimum observations required (default: 1)
    
    Returns:
        pd.Series: Series of rolling volatility values
    """
    # Implement rolling volatility calculation using returns
    returns = prices.pct_change()
    rolling_volatility = returns.rolling(window=window, min_periods=min_periods).std()
    return rolling_volatility


def calculate_z_scores(data: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Z-scores for detecting anomalies in volatility.
    
    Z-score = (value - mean) / std_dev
    
    Args:
        data (pd.Series): Series of data points to analyze
        window (int): Rolling window for mean and std calculation
    
    Returns:
        pd.Series: Series of Z-score values
    """
    # Implement Z-score calculation with rolling statistics
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    z_scores = (data - rolling_mean) / rolling_std
    return z_scores


def identify_volatility_events(
    volatility: pd.Series,
    z_scores: pd.Series,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Identify periods of significant volatility events.
    
    Args:
        volatility (pd.Series): Rolling volatility values
        z_scores (pd.Series): Z-score values
        threshold (float): Z-score threshold for event detection (default: 2.0)
    
    Returns:
        pd.DataFrame: DataFrame with event dates and volatility metrics
    """
    # Implement event detection logic based on Z-score threshold
    events = z_scores[z_scores.abs() > threshold]
    event_df = pd.DataFrame({
        'Date': events.index,
        'Volatility': volatility[events.index],
        'Z-Score': events
    })
    return event_df
