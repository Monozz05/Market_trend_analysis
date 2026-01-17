"""
Main Streamlit Application

Interactive dashboard for Event-Driven Market Volatility Analysis.

Features:
- Company selector dropdown (TSLA, AAPL, AMZN)
- Interactive volatility analysis visualization
- Event detection and impact analysis
- Text-based insights and recommendations
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src and parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_stock_data
from src.volatility import calculate_volatility
from src.event_analysis import analyze_event_impact
from src.insights import generate_insights


def load_events_data():
    """
    Load events data from CSV file.
    
    Returns:
        pd.DataFrame: Events DataFrame or None if file not found
    """
    try:
        events_path = Path(__file__).parent.parent / "data" / "events.csv"
        if events_path.exists():
            events_df = pd.read_csv(events_path)
            return events_df
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load events.csv: {e}")
        return None


def main():
    """Main Streamlit application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Event-Driven Market Volatility Analysis",
        page_icon="�",
        layout="wide"
    )
    
    # Header
    st.title("Event-Driven Market Volatility Analysis")
    st.markdown("""
    Analyze stock price volatility using rolling statistics and Z-score normalization.
    Detect high-volatility periods and assess the impact of market events.
    """)
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    
    # Company selection dropdown
    # Available companies: TSLA, AAPL, AMZN (as specified in requirements)
    available_companies = ['TSLA', 'AAPL', 'AMZN']
    
    selected_company = st.sidebar.selectbox(
        "Select Company",
        options=available_companies,
        help="Choose a company to analyze"
    )
    
    # Run Analysis button
    analyze_button = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)
    
    # Information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Analysis Pipeline:**
    1. Load historical stock data (1 year)
    2. Calculate 7-day rolling statistics
    3. Compute Z-scores for volatility detection
    4. Analyze event impact (7 days before/after)
    5. Generate quantitative insights
    """)
    
    # Main content area
    if analyze_button:
        try:
            # === STEP 1: LOAD STOCK DATA ===
            with st.spinner(f"Loading {selected_company} data..."):
                df = load_stock_data(selected_company)
                st.success(f"Loaded {len(df)} days of {selected_company} stock data")
            
            # === STEP 2: LOAD EVENTS DATA ===
            with st.spinner("Loading events data..."):
                events_df = load_events_data()
                if events_df is not None:
                    # Filter events for selected company
                    company_events = events_df[events_df['Company'] == selected_company]
                    st.success(f"Found {len(company_events)} event(s) for {selected_company}")
                else:
                    company_events = pd.DataFrame()
                    st.warning("No events.csv found - continuing without event analysis")
            
            # === STEP 3: CALCULATE VOLATILITY ===
            with st.spinner("Calculating volatility metrics..."):
                df = calculate_volatility(df)
                st.success("Volatility metrics calculated")
            
            # === STEP 4: RUN EVENT IMPACT ANALYSIS ===
            with st.spinner("Analyzing event impact..."):
                if not company_events.empty:
                    event_summary = analyze_event_impact(df, company_events)
                else:
                    event_summary = []
                st.success("Event analysis complete")
            
            # === STEP 5: GENERATE INSIGHTS ===
            with st.spinner("Generating insights..."):
                insights_text = generate_insights(df, event_summary)
                st.success("Insights generated")
            
            # === DISPLAY RESULTS ===
            st.markdown("---")
            st.header("Analysis Results")
            
            # Display insights in an info box
            st.markdown("### Key Insights")
            st.markdown(insights_text)
            
            st.markdown("---")
            
            # === VISUALIZATIONS ===
            st.markdown("### Price & Volatility Analysis with Event Impact")
            
            # Prepare event data for plotting
            event_dates = []
            event_labels = []
            event_colors = []
            
            if event_summary:
                for event in event_summary:
                    event_date = pd.to_datetime(event['date'])
                    if event_date >= df['Date'].min() and event_date <= df['Date'].max():
                        event_dates.append(event_date)
                        event_labels.append(event['event'])
                        # Color by impact
                        if 'Impact' in company_events.columns:
                            impact_type = company_events[company_events['Event'] == event['event']]['Impact'].values
                            if len(impact_type) > 0:
                                if impact_type[0] == 'Positive':
                                    event_colors.append('green')
                                elif impact_type[0] == 'Negative':
                                    event_colors.append('red')
                                else:
                                    event_colors.append('orange')
                            else:
                                event_colors.append('blue')
                        else:
                            event_colors.append('blue')
            
            # Chart 1: Price with Event Markers
            st.subheader("Stock Price with Event Impact")
            st.markdown("**Green** = Positive events | **Red** = Negative events | **Orange** = Neutral events")
            
            fig1, ax1 = plt.subplots(figsize=(14, 6))
            
            # Plot price line
            ax1.plot(df['Date'], df['Close'], color='#1f77b4', linewidth=2.5, label='Close Price', zorder=1)
            ax1.fill_between(df['Date'], df['Close'], alpha=0.2, color='#1f77b4')
            
            # Add event markers
            for i, (event_date, label, color) in enumerate(zip(event_dates, event_labels, event_colors)):
                # Find closest price
                closest_idx = (df['Date'] - event_date).abs().argmin()
                price = df.iloc[closest_idx]['Close']
                
                # Add vertical line
                ax1.axvline(x=event_date, color=color, linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                # Add marker
                ax1.scatter(event_date, price, color=color, s=200, marker='o', 
                           edgecolor='black', linewidth=2, zorder=3, label=label if i < 3 else "")
                
                # Add annotation
                ax1.annotate(f'{i+1}', xy=(event_date, price), 
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8, edgecolor='black'),
                           color='white', zorder=4)
            
            ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
            ax1.set_title(f'{selected_company} - Stock Price with Event Timeline', 
                         fontsize=15, fontweight='bold', pad=20)
            ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig1)
            
            # Event Legend
            if event_summary:
                st.markdown("**Event Legend:**")
                for i, event in enumerate(event_summary):
                    color_indicator = "+" if event_colors[i] == 'green' else "-" if event_colors[i] == 'red' else "~"
                    st.markdown(f"**[{i+1}]** {event['event']} ({event['date']}) `{color_indicator}`")
            
            st.markdown("---")
            
            # Chart 2: Before/After Price Comparison
            st.subheader("Price Change Analysis: Before vs After Events")
            
            if event_summary:
                fig2, axes = plt.subplots(1, len(event_summary), figsize=(6*len(event_summary), 5))
                if len(event_summary) == 1:
                    axes = [axes]
                
                for idx, event in enumerate(event_summary):
                    event_date = pd.to_datetime(event['date'])
                    
                    # Get prices before and after
                    event_position = (df['Date'] - event_date).abs().argmin()
                    
                    # 5 days before and after
                    start_idx = max(0, event_position - 5)
                    end_idx = min(len(df), event_position + 6)
                    
                    before_prices = df.iloc[start_idx:event_position]['Close'].values
                    after_prices = df.iloc[event_position+1:end_idx]['Close'].values
                    
                    if len(before_prices) > 0 and len(after_prices) > 0:
                        before_avg = before_prices.mean()
                        after_avg = after_prices.mean()
                        price_change_pct = ((after_avg - before_avg) / before_avg) * 100
                        
                        # Bar chart
                        bars = axes[idx].bar(['Before Event\n(5 days)', 'After Event\n(5 days)'], 
                                            [before_avg, after_avg],
                                            color=[event_colors[idx] if event_colors[idx] != 'orange' else 'gray', 
                                                   event_colors[idx]],
                                            alpha=0.7, edgecolor='black', linewidth=2)
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                                         f'${height:.2f}',
                                         ha='center', va='bottom', fontsize=11, fontweight='bold')
                        
                        # Add change annotation
                        arrow_color = 'green' if price_change_pct > 0 else 'red'
                        arrow = '↑' if price_change_pct > 0 else '↓'
                        axes[idx].text(0.5, max(before_avg, after_avg) * 1.1,
                                      f'{arrow} {abs(price_change_pct):.2f}%',
                                      ha='center', fontsize=13, fontweight='bold',
                                      color=arrow_color,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        axes[idx].set_ylabel('Average Price ($)', fontsize=11, fontweight='bold')
                        axes[idx].set_title(f'{event["event"][:30]}...', 
                                          fontsize=10, fontweight='bold', wrap=True)
                        axes[idx].grid(True, alpha=0.3, axis='y')
                        axes[idx].set_ylim(0, max(before_avg, after_avg) * 1.2)
                
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No events to display for comparison")
            
            st.markdown("---")
            
            # Chart 3: Z-Score with Event Markers
            st.subheader("Volatility Z-Score with Event Impact")
            st.markdown("""
            **Z-Score Interpretation:**
            - **|Z-score| < 1:** Normal volatility
            - **1 < |Z-score| < 2:** Moderate volatility
            - **|Z-score| > 2:** High volatility (price significantly deviated)
            """)
            
            fig3, ax3 = plt.subplots(figsize=(14, 6))
            
            # Plot Z-score line
            ax3.plot(df['Date'], df['z_score'], color='#ff7f0e', linewidth=2.5, label='Z-Score', zorder=1)
            
            # Add horizontal reference lines
            ax3.axhline(y=2, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='High volatility threshold')
            ax3.axhline(y=-2, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
            ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Mean')
            
            # Fill high volatility regions
            ax3.fill_between(df['Date'], df['z_score'], 2, where=(df['z_score'] >= 2), 
                            color='red', alpha=0.2, interpolate=True, label='High volatility zone')
            ax3.fill_between(df['Date'], df['z_score'], -2, where=(df['z_score'] <= -2), 
                            color='red', alpha=0.2, interpolate=True)
            
            # Add event markers
            for i, (event_date, label, color) in enumerate(zip(event_dates, event_labels, event_colors)):
                # Find closest z-score
                closest_idx = (df['Date'] - event_date).abs().argmin()
                z_score = df.iloc[closest_idx]['z_score']
                
                # Add vertical line
                ax3.axvline(x=event_date, color=color, linestyle='--', alpha=0.7, linewidth=2, zorder=2)
                
                # Add marker
                ax3.scatter(event_date, z_score, color=color, s=200, marker='o',
                           edgecolor='black', linewidth=2, zorder=3)
                
                # Add number label
                ax3.annotate(f'{i+1}', xy=(event_date, z_score),
                           xytext=(0, 15), textcoords='offset points',
                           ha='center', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8, edgecolor='black'),
                           color='white', zorder=4)
            
            ax3.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax3.set_ylabel('Z-Score', fontsize=13, fontweight='bold')
            ax3.set_title(f'{selected_company} - Volatility Z-Score with Event Timeline',
                         fontsize=15, fontweight='bold', pad=20)
            ax3.grid(True, alpha=0.3, linestyle=':', linewidth=1)
            ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
            
            st.markdown("---")
            
            # === EVENT ANALYSIS DETAILS ===
            if event_summary:
                st.markdown("### Detailed Event Impact Metrics")
                
                # Convert to DataFrame for display
                events_display = pd.DataFrame(event_summary)
                
                # Rename columns for better display
                display_columns = {
                    'date': 'Date',
                    'event': 'Event',
                    'price_before': 'Avg Price Before ($)',
                    'price_after': 'Avg Price After ($)',
                    'price_change_pct': 'Price Change (%)',
                    'z_score_before': 'Z-Score Before',
                    'z_score_after': 'Z-Score After',
                    'impact': 'Volatility Impact',
                    'impact_direction': 'Direction',
                    'accuracy_score': 'Confidence (%)'
                }
                
                events_display = events_display.rename(columns=display_columns)
                
                # Reorder columns for better readability
                column_order = [
                    'Date', 'Event', 
                    'Avg Price Before ($)', 'Avg Price After ($)', 'Price Change (%)',
                    'Z-Score Before', 'Z-Score After', 'Volatility Impact',
                    'Direction', 'Confidence (%)'
                ]
                events_display = events_display[column_order]
                
                # Style the dataframe
                def highlight_price_change(val):
                    if isinstance(val, (int, float)):
                        color = 'lightgreen' if val > 0 else 'lightcoral' if val < 0 else 'white'
                        return f'background-color: {color}'
                    return ''
                
                # Display styled dataframe
                styled_df = events_display.style.applymap(
                    highlight_price_change,
                    subset=['Price Change (%)']
                ).format({
                    'Avg Price Before ($)': '${:.2f}',
                    'Avg Price After ($)': '${:.2f}',
                    'Price Change (%)': '{:+.2f}%',
                    'Z-Score Before': '{:.3f}',
                    'Z-Score After': '{:.3f}',
                    'Volatility Impact': '{:+.3f}',
                    'Confidence (%)': '{:.1f}%'
                })
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                st.markdown("""
                **How to read this table:**
                
                **Price Metrics:**
                - **Avg Price Before/After:** Average closing price in the 5 days before/after event
                - **Price Change (%):** Actual percentage change in stock price
                  - Positive values = Price increased after event
                  - Negative values = Price decreased after event
                
                **Volatility Metrics:**
                - **Z-Score Before/After:** Average volatility measure 5 days before/after
                - **Volatility Impact:** Change in volatility (Z-score difference)
                - **Direction:** Whether volatility increased or decreased
                
                **Confidence (%):** Statistical confidence in the measurement
                - Based on data availability, sample size, and impact magnitude
                - Higher percentage = more reliable measurement
                """)
                
                # Summary statistics
                st.markdown("### Event Impact Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                avg_price_change = events_display['Price Change (%)'].mean()
                max_price_change = events_display['Price Change (%)'].max()
                avg_volatility_impact = events_display['Volatility Impact'].abs().mean()
                avg_confidence = events_display['Confidence (%)'].mean()
                
                with col1:
                    st.metric(
                        "Avg Price Impact",
                        f"{avg_price_change:+.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Max Price Change",
                        f"{max_price_change:+.2f}%",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Avg Volatility Impact",
                        f"{avg_volatility_impact:.3f}",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Avg Confidence",
                        f"{avg_confidence:.1f}%",
                        delta=None
                    )
            
            st.markdown("---")
            
            # === DATA PREVIEW ===
            with st.expander("View Raw Data"):
                st.dataframe(df, use_container_width=True)
            
        except FileNotFoundError as e:
            st.error(f"**Error:** {e}")
            st.info("Make sure the CSV file exists in the `data/` directory")
            
        except ValueError as e:
            st.error(f"**Data validation error:** {e}")
            st.info("Check that your CSV file has the required columns")
            
        except Exception as e:
            st.error(f"**An unexpected error occurred:** {e}")
            st.exception(e)
    
    else:
        # Welcome screen when no analysis is running
        st.info("Select a company from the sidebar and click **Run Analysis** to begin")
        
        # Show example of what will be analyzed
        st.markdown("### System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Volatility Analysis**
            - 7-day rolling mean
            - Rolling std deviation
            - Z-score calculation
            """)
        
        with col2:
            st.markdown("""
            **Event Impact**
            - Before/after comparison
            - 5-day windows
            - Impact quantification
            """)
        
        with col3:
            st.markdown("""
            **Insights**
            - High-volatility days
            - Price trends
            - Event correlations
            """)
        
        st.markdown("---")
        st.markdown("""
        **Available Companies:** TSLA, AAPL, AMZN  
        **Data Period:** Last 1 year from the most recent date in each dataset  
        **Technology:** Pandas, NumPy, Streamlit, Matplotlib
        """)


if __name__ == "__main__":
    main()
