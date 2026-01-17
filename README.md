# Event-Driven Market Volatility Analysis

## Project Summary

This project quantifies the impact of specific historical events on stock market volatility using transparent, explainable statistical methods. The system analyzes historical stock price data for Tesla (TSLA), Apple (AAPL), and Amazon (AMZN), correlating price movements with real-world events such as earnings announcements, product launches, and leadership transitions.

The approach emphasizes interpretability over complexity, using rolling statistics and Z-score normalization to detect anomalous price behavior. Each analysis includes confidence scores derived from data availability, impact magnitude, and price consistency metrics. This ensures that every result is auditable and transparent.

## Selected Track

**AI for Market Trend Analysis**

## Key Features

- Multi-company analysis across three major technology stocks
- Event-driven volatility detection using 7-day rolling windows
- Z-score normalization for statistical anomaly identification
- Confidence scoring algorithm for result reliability quantification
- Before-after event comparison with percentage price changes
- Human-readable insight generation from numerical analysis
- Interactive Streamlit web interface with color-coded visualizations
- Comprehensive Jupyter Notebook for reproducible analysis

## Project Structure

```
Market_trend_analysis/
├── app/
│   ├── __init__.py
│   └── main.py                                    # Streamlit web application
├── data/
│   ├── TSLA.csv                                   # Tesla stock data (252 days)
│   ├── AAPL.csv                                   # Apple stock data (506 days)
│   ├── AMZN.csv                                   # Amazon stock data (255 days)
│   └── events.csv                                 # Historical event dataset (12 events)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                             # CSV loading with format validation
│   ├── volatility.py                              # Rolling statistics and Z-score calculation
│   ├── event_analysis.py                          # Event impact quantification
│   └── insights.py                                # Text-based insight generation
├── Event_Driven_Market_Volatility_Analysis.ipynb  # Primary evaluation notebook
├── requirements.txt                               # Python dependencies
├── .gitignore
└── README.md
```

## Data Description

**Stock Price Data:**
- Tesla (TSLA): 252 trading days from February 2024 to February 2025
- Apple (AAPL): 506 trading days from January 2018 to December 2020
- Amazon (AMZN): 255 trading days from December 2020 to December 2021
- Columns: Date, Open, High, Low, Close, Volume

**Event Dataset:**
- 12 historical events across three companies
- Event types: quarterly earnings reports, product launches, CEO transitions, shareholder meetings
- Each event includes date, company, and description
- Events are cross-validated with actual historical dates

**Data Source:**
Publicly available datasets from Kaggle and financial data repositories.

## How to Run

### Environment Setup

1. Clone or download the repository
2. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run Streamlit Web Application

Launch the interactive web interface:
```bash
streamlit run app/main.py
```
The application will open in your default browser at `http://localhost:8501`

Features:
- Company selection dropdown
- Three visualization charts with event markers
- Event metrics table with confidence scores
- Generated insights summary

### Run Jupyter Notebook

For detailed analysis and reproducible results:
```bash
jupyter notebook Event_Driven_Market_Volatility_Analysis.ipynb
```
Or open the notebook in VS Code with Jupyter extension installed.

The notebook contains:
- Complete data loading for all companies
- Step-by-step volatility calculations
- Event impact analysis with detailed metrics
- Comparative visualizations
- Statistical evaluation and interpretation

## Evaluation Note

**The Jupyter Notebook (Event_Driven_Market_Volatility_Analysis.ipynb) is the primary source of truth for evaluation.**

The notebook follows a structured 10-section format covering:
1. Project introduction and objectives
2. Problem definition and motivation
3. Data understanding and statistics
4. System architecture and design principles
5. Complete implementation with code execution
6. Results evaluation and interpretation
7. Ethical considerations and limitations
8. Conclusions and future scope

All cells are executable top-to-bottom without errors. The notebook reuses modular code from the `src/` directory to ensure consistency with the Streamlit application.

## Methodology

### Statistical Approach

**Volatility Detection:**
- 7-day rolling window for mean and standard deviation calculation
- Z-score normalization: Z = (Price - Rolling Mean) / Rolling Standard Deviation
- Threshold: |Z| > 2 indicates high volatility (95% confidence level)

**Event Impact Quantification:**
- Compare average prices 7 days before vs. 7 days after event
- Calculate percentage price change
- Measure volatility spike during event window

**Confidence Scoring:**
The system computes a 0-100% confidence score for each event analysis:
- Data Availability (50% weight): Sufficient data points before and after event
- Impact Magnitude (30% weight): Significance of price change
- Price Consistency (20% weight): Stability of before/after windows

### Explainability Principles

- All calculations use standard statistical methods (mean, standard deviation, Z-score)
- No black-box machine learning models
- Every metric includes confidence indicators
- Results are translated into human-readable insights
- Complete audit trail from raw data to conclusions

## Ethical Considerations

**This project adheres to strict ethical guidelines:**

1. **No Financial Advice**: This system is designed for educational and research purposes only. It does not provide investment recommendations or trading signals.

2. **No Profit Prediction**: The analysis examines historical correlations between events and volatility. It makes no claims about predicting future stock prices or market movements.

3. **Transparent Logic**: All methodologies are documented and reproducible. The statistical approach is based on interpretable techniques, not opaque algorithms.

4. **Known Limitations**:
   - Correlation does not imply causation
   - Historical patterns may not repeat
   - Analysis covers limited time periods and companies
   - Market movements often have multiple simultaneous causes

5. **Responsible Use**: Results should be interpreted as academic analysis of past events, not as guidance for financial decisions.

## Future Scope

Potential enhancements for extending this work:

- **Automated Event Extraction**: Use natural language processing to identify events from financial news APIs and regulatory filings
- **Multi-Sector Analysis**: Expand coverage to healthcare, energy, finance, and consumer goods sectors
- **Real-Time Monitoring**: Implement streaming data pipelines for live volatility detection
- **Enhanced Visualizations**: Add interactive time-series sliders, event filtering, and comparative dashboards
- **Sentiment Integration**: Correlate social media sentiment scores with price movements
- **Benchmark Comparison**: Validate results against established financial models (GARCH, VaR)

## Technical Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt:
  - streamlit >= 1.30.0
  - pandas >= 2.0.0
  - numpy >= 1.24.0
  - matplotlib >= 3.7.0

## License

This project is submitted for academic evaluation.