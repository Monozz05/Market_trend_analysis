"""
Event Impact Analysis Module

Responsible for analyzing the impact of detected events on market behavior.

This module provides functionality to:
- Quantify event magnitude and duration
- Analyze price movement during events
- Correlate events with other metrics
"""

import pandas as pd
import numpy as np


def analyze_event_impact(df: pd.DataFrame, events_df: pd.DataFrame) -> list:
    """
    Analyze the impact of events on stock volatility and price.
    
    This function:
    - Loads events from events.csv with Date column parsed
    - For each event, compares Z-scores AND prices before and after
    - Calculates mean Z-score for 5 days before vs 5 days after event
    - Calculates actual price change percentage
    - Returns a structured summary of event impacts with accurate metrics
    
    Logic:
    - If Z-score increases after event: volatility increased (higher impact)
    - If Z-score decreases after event: volatility decreased (stabilizing)
    - Price change percentage shows actual financial impact
    - Difference > 0.5: noticeable impact
    
    Args:
        df (pd.DataFrame): Stock data DataFrame with 'Date', 'Close', and 'z_score' columns
        events_df (pd.DataFrame): Events DataFrame with 'Date', 'Event', 'Company' columns
    
    Returns:
        list: List of dictionaries with event impact analysis
              Each dict contains: {
                  'date': event date,
                  'event': event description,
                  'z_score_before': mean z-score 5 days before,
                  'z_score_after': mean z-score 5 days after,
                  'impact': difference (after - before),
                  'impact_direction': 'increased' or 'decreased',
                  'price_before': average price 5 days before,
                  'price_after': average price 5 days after,
                  'price_change_pct': percentage change in price,
                  'accuracy_score': confidence score (0-100%)
              }
    """
    # Initialize results list
    event_impacts = []
    
    # Ensure Date column is datetime in events_df
    if 'Date' not in events_df.columns:
        return event_impacts  # Return empty if no Date column
    
    # Parse event dates as datetime
    events_df = events_df.copy()
    events_df['Date'] = pd.to_datetime(events_df['Date'], format='mixed', dayfirst=True)
    
    # Ensure df has Date as datetime
    if 'Date' not in df.columns:
        return event_impacts
    
    # Set Date as index for easier time-based filtering
    df = df.set_index('Date')
    
    # Analyze each event
    for idx, event_row in events_df.iterrows():
        event_date = event_row['Date']
        event_name = event_row.get('Event', 'Unknown Event')
        
        # Skip if event date is not in our data range
        if event_date not in df.index:
            # Find closest date in range
            if event_date < df.index.min() or event_date > df.index.max():
                continue
        
        # Calculate date windows: 5 days before and 5 days after
        # We use business days approach - just take 5 days of actual data
        
        try:
            # Get position of event date in the DataFrame
            event_position = df.index.get_indexer([event_date], method='nearest')[0]
            
            # Get 5 days before (5 rows before event)
            start_before = max(0, event_position - 5)
            end_before = event_position
            
            # Get 5 days after (5 rows after event)
            start_after = event_position + 1
            end_after = min(len(df), event_position + 6)
            
            # Extract Z-scores for before and after periods
            z_scores_before = df.iloc[start_before:end_before]['z_score'].dropna()
            z_scores_after = df.iloc[start_after:end_after]['z_score'].dropna()
            
            # Extract prices for before and after periods
            prices_before = df.iloc[start_before:end_before]['Close'].dropna()
            prices_after = df.iloc[start_after:end_after]['Close'].dropna()
            
            # Calculate mean Z-scores (handle empty arrays)
            if len(z_scores_before) > 0:
                mean_z_before = z_scores_before.mean()
            else:
                mean_z_before = 0.0
            
            if len(z_scores_after) > 0:
                mean_z_after = z_scores_after.mean()
            else:
                mean_z_after = 0.0
            
            # Calculate mean prices
            if len(prices_before) > 0:
                mean_price_before = prices_before.mean()
            else:
                mean_price_before = 0.0
            
            if len(prices_after) > 0:
                mean_price_after = prices_after.mean()
            else:
                mean_price_after = 0.0
            
            # Calculate impact: difference in average Z-score
            impact = mean_z_after - mean_z_before
            
            # Calculate price change percentage
            if mean_price_before > 0:
                price_change_pct = ((mean_price_after - mean_price_before) / mean_price_before) * 100
            else:
                price_change_pct = 0.0
            
            # Determine impact direction
            if impact > 0:
                impact_direction = 'increased'
            elif impact < 0:
                impact_direction = 'decreased'
            else:
                impact_direction = 'no change'
            
            # Calculate accuracy score (confidence metric)
            # Based on: data availability, sample size, and consistency
            data_availability = (len(z_scores_before) + len(z_scores_after)) / 10 * 50  # Max 50%
            impact_magnitude = min(abs(impact) / 3, 1) * 30  # Max 30%
            price_consistency = min(abs(price_change_pct) / 10, 1) * 20  # Max 20%
            accuracy_score = min(data_availability + impact_magnitude + price_consistency, 100)
            
            # Store event impact summary
            event_impacts.append({
                'date': event_date.strftime('%Y-%m-%d'),
                'event': event_name,
                'z_score_before': round(mean_z_before, 3),
                'z_score_after': round(mean_z_after, 3),
                'impact': round(impact, 3),
                'impact_direction': impact_direction,
                'price_before': round(mean_price_before, 2),
                'price_after': round(mean_price_after, 2),
                'price_change_pct': round(price_change_pct, 2),
                'accuracy_score': round(accuracy_score, 1)
            })
            
        except Exception as e:
            # If any error occurs for this event, skip it
            continue
    
    return event_impacts


def analyze_event_magnitude(
    volatility_events: pd.DataFrame,
    prices: pd.Series
) -> pd.DataFrame:
    """
    Analyze the magnitude and impact of volatility events.
    
    Args:
        volatility_events (pd.DataFrame): DataFrame of detected events
        prices (pd.Series): Price data for correlation analysis
    
    Returns:
        pd.DataFrame: Enhanced event DataFrame with magnitude metrics
    """
    # Implement event magnitude calculation
    # Example: Calculate the average price change during each event
    event_magnitudes = []
    for index, event in volatility_events.iterrows():
        start_date = event['start_date']
        end_date = event['end_date']
        event_prices = prices[start_date:end_date]
        magnitude = (event_prices.iloc[-1] - event_prices.iloc[0]) / event_prices.iloc[0]
        event_magnitudes.append(magnitude)
    
    volatility_events['magnitude'] = event_magnitudes
    return volatility_events


def calculate_event_duration(volatility_events: pd.DataFrame) -> pd.Series:
    """
    Calculate the duration (in days) of each volatility event.
    
    Args:
        volatility_events (pd.DataFrame): DataFrame of detected events
    
    Returns:
        pd.Series: Duration of each event
    """
    # Implement event duration calculation
    # Example: Calculate the number of days between start and end date of each event
    durations = (volatility_events['end_date'] - volatility_events['start_date']).dt.days
    return durations


def get_price_changes_during_events(
    prices: pd.Series,
    events: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate price changes and percentage returns during event periods.
    
    Args:
        prices (pd.Series): Historical price data
        events (pd.DataFrame): DataFrame of event periods
    
    Returns:
        pd.DataFrame: Price changes and returns during events
    """
    # Implement price change calculation during events
    # Example: Calculate the price change and percentage return for each event period
    price_changes = []
    percentage_returns = []
    
    for index, event in events.iterrows():
        start_date = event['start_date']
        end_date = event['end_date']
        start_price = prices[start_date]
        end_price = prices[end_date]
        price_change = end_price - start_price
        percentage_return = (price_change / start_price) * 100
        
        price_changes.append(price_change)
        percentage_returns.append(percentage_return)
    
    result_df = pd.DataFrame({
        'price_change': price_changes,
        'percentage_return': percentage_returns
    })
    return result_df
