"""
Insight Generation Module

Responsible for generating human-readable insights from the analysis.

This module provides functionality to:
- Summarize key findings
- Generate textual analysis
- Create actionable recommendations
"""

import pandas as pd


def generate_insights(df: pd.DataFrame, event_summary: list) -> str:
    """
    Generate clear, human-readable text insights from the analysis.
    
    This function creates a comprehensive text summary including:
    - Data date range covered
    - Number of high-volatility days (|z_score| > 2)
    - Events with noticeable impact on volatility
    - Overall volatility assessment
    
    Args:
        df (pd.DataFrame): Stock data DataFrame with Date, Close, and z_score columns
        event_summary (list): List of event impact dictionaries from analyze_event_impact()
    
    Returns:
        str: Multi-line text string with human-readable insights
    """
    insights = []
    
    # === DATA RANGE INSIGHT ===
    if 'Date' in df.columns:
        start_date = df['Date'].min().strftime('%B %d, %Y')
        end_date = df['Date'].max().strftime('%B %d, %Y')
        insights.append(f"**Analysis Period:** {start_date} to {end_date}")
        insights.append(f"**Total Trading Days:** {len(df)}")
    
    # === VOLATILITY INSIGHTS ===
    if 'z_score' in df.columns:
        # Count high-volatility days (absolute Z-score > 2)
        # Z-score > 2 means price deviated significantly from recent average
        high_volatility_days = df[df['z_score'].abs() > 2].shape[0]
        
        # Calculate percentage
        total_days = len(df[df['z_score'].notna()])
        if total_days > 0:
            volatility_percentage = (high_volatility_days / total_days) * 100
            insights.append(f"\n**High Volatility Days:** {high_volatility_days} days ({volatility_percentage:.1f}%)")
            insights.append(f"   (Days where |Z-score| > 2, indicating significant price deviation)")
        
        # Overall volatility assessment
        if volatility_percentage > 20:
            insights.append(f"   **Assessment:** High volatility period - prices were unstable")
        elif volatility_percentage > 10:
            insights.append(f"   **Assessment:** Moderate volatility - some price fluctuations")
        else:
            insights.append(f"   **Assessment:** Low volatility - relatively stable prices")
    
    # === PRICE INSIGHTS ===
    if 'Close' in df.columns:
        first_price = df['Close'].iloc[0]
        last_price = df['Close'].iloc[-1]
        price_change = last_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        insights.append(f"\n**Price Movement:**")
        insights.append(f"   Starting Price: ${first_price:.2f}")
        insights.append(f"   Ending Price: ${last_price:.2f}")
        
        if price_change > 0:
            insights.append(f"   Change: +${price_change:.2f} (+{price_change_pct:.2f}%)")
        else:
            insights.append(f"   Change: ${price_change:.2f} ({price_change_pct:.2f}%)")
    
    # === EVENT IMPACT INSIGHTS ===
    if event_summary and len(event_summary) > 0:
        insights.append(f"\n**Event Impact Analysis:**")
        insights.append(f"   Analyzed {len(event_summary)} event(s)")
        
        # Find events with noticeable impact (|impact| > 0.5)
        noticeable_events = [e for e in event_summary if abs(e['impact']) > 0.5]
        
        if noticeable_events:
            insights.append(f"\n   **Events with Noticeable Impact:**")
            for event in noticeable_events:
                impact_dir = "increase" if event['impact'] > 0 else "decrease"
                price_dir = "gain" if event['price_change_pct'] > 0 else "loss"
                
                insights.append(f"   **{event['event']}** ({event['date']})")
                insights.append(f"      - Z-score impact: {event['impact']:+.2f} ({event['impact_direction']})")
                insights.append(f"      - Price change: {event['price_change_pct']:+.2f}% ({price_dir})")
                insights.append(f"      - Avg price before: ${event['price_before']:.2f}, after: ${event['price_after']:.2f}")
                insights.append(f"      - Confidence: {event['accuracy_score']:.1f}%")
        else:
            insights.append(f"   No events showed significant volatility impact (threshold: 0.5)")
        
        # Summary statistics
        avg_impact = sum([abs(e['impact']) for e in event_summary]) / len(event_summary)
        avg_price_change = sum([abs(e['price_change_pct']) for e in event_summary]) / len(event_summary)
        insights.append(f"\n   **Overall Event Statistics:**")
        insights.append(f"   - Average volatility impact: {avg_impact:.2f}")
        insights.append(f"   - Average price change: {avg_price_change:.2f}%")
        
    else:
        insights.append(f"\n**Event Impact Analysis:**")
        insights.append(f"   No matching events found for this company in the analysis period")
    
    # === SUMMARY ===
    insights.append(f"\n**Summary:**")
    insights.append(f"   This analysis used 7-day rolling statistics to detect volatility patterns")
    insights.append(f"   and compared market behavior before/after key events.")
    
    # Join all insights with newlines
    return '\n'.join(insights)


def generate_summary_statistics(
    volatility: pd.Series,
    events: pd.DataFrame,
    ticker: str
) -> dict:
    """
    Generate summary statistics for the analysis period.
    
    Args:
        volatility (pd.Series): Volatility data
        events (pd.DataFrame): Detected events
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Dictionary with key statistics
    """
    # TODO: Implement summary statistics generation
    pass


def generate_text_insights(
    summary_stats: dict,
    events: pd.DataFrame,
    ticker: str
) -> str:
    """
    Generate human-readable text insights from the analysis.
    
    Args:
        summary_stats (dict): Summary statistics dictionary
        events (pd.DataFrame): Detected events
        ticker (str): Stock ticker symbol
    
    Returns:
        str: Formatted text insights
    """
    # TODO: Implement text insight generation
    pass


def create_recommendations(insights: str) -> list:
    """
    Create actionable recommendations based on analysis insights.
    
    Args:
        insights (str): Text insights from analysis
    
    Returns:
        list: List of recommendations
    """
    # TODO: Implement recommendation logic
    pass
