# FinBridge Pro - Advanced Financial Analytics Platform

A comprehensive financial analytics and market screener platform with advanced technical analysis, risk metrics, and portfolio optimization tools.

## 🚀 Features

### 📊 Stock Analysis
- **Real-time Stock Data**: Live price charts with candlestick patterns
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic, ATR, ADX, CCI, Williams %R, ROC, OBV, VWAP
- **Support & Resistance**: Automatic calculation of key price levels
- **Fibonacci Retracements**: Golden ratio analysis for trend reversals
- **Risk Metrics**: VaR, Max Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Skewness, Kurtosis

### 🔍 Market Screener
- **Advanced Filtering**: Price, Market Cap, P/E Ratio, Dividend Yield, Beta, Volume
- **Market Cap Categories**: Micro, Small, Mid, Large, Mega Cap stocks
- **Sorting Options**: Multiple criteria for ranking stocks
- **Real-time Results**: Instant screening with actionable insights

### 📈 Financial Models
- **CAPM (Capital Asset Pricing Model)**: Expected return calculation
- **Monte Carlo Simulation**: Price forecasting with multiple scenarios
- **Black-Scholes Options Pricing**: Call and Put option valuation
- **Fama-French 3-Factor Model**: Multi-factor risk analysis
- **Modern Portfolio Theory (MPT)**: Portfolio optimization and efficient frontier

### 🎯 Sentiment Analysis
- **Technical Sentiment**: Based on multiple technical indicators
- **Signal Generation**: Buy/Sell/Hold recommendations
- **Confidence Scoring**: Probability-based analysis
- **Market Sentiment**: Overall market direction analysis

### 📋 Comprehensive Reports
- **Stock Analysis Reports**: Complete fundamental and technical analysis
- **Performance Metrics**: Annual returns, volatility, risk-adjusted returns
- **Trading Signals**: Trend, momentum, volume, and volatility analysis
- **Recommendations**: Actionable investment advice

### 🌍 Market Overview
- **Multi-Market Support**: NYSE and PSX markets
- **Top Gainers/Losers**: Real-time market movers
- **Most Active Stocks**: Volume-based analysis
- **Market Sentiment**: Overall market direction

### ⭐ Watchlist Management
- **Personal Watchlists**: Track favorite stocks
- **Notes & Alerts**: Custom annotations for stocks
- **Quick Analysis**: One-click comprehensive analysis

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **Data**: yfinance API for real-time stock data
- **Technical Analysis**: TA-Lib library
- **Machine Learning**: scikit-learn for advanced analytics
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Charts**: Plotly.js for interactive visualizations
- **UI Framework**: Bootstrap 5
- **Icons**: Font Awesome

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Financial-project
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```

6. **Access the platform**:
   Open your browser and go to `http://localhost:5000`

## 🎮 Usage Guide

### Getting Started
1. **Login/Register**: Use any credentials for demo access
2. **Select Market**: Choose between NYSE or PSX
3. **Select Stock**: Pick from available stocks
4. **Load Data**: Click "Load Data" to fetch stock information

### Stock Analysis
1. **View Charts**: Interactive candlestick charts with technical indicators
2. **Technical Analysis**: Click "Technical Analysis" for RSI, MACD, and more
3. **Risk Metrics**: Click "Risk Metrics" for comprehensive risk analysis
4. **Generate Report**: Get complete analysis with recommendations

### Market Screening
1. **Set Criteria**: Define price, market cap, P/E ratio filters
2. **Run Screener**: Click "Screen Market" to find matching stocks
3. **Analyze Results**: Click "Analyze" on any stock for detailed analysis
4. **Add to Watchlist**: Save interesting stocks for later review

### Financial Models
1. **CAPM**: Enter beta, market return, and risk-free rate
2. **Monte Carlo**: Set parameters for price simulation
3. **Black-Scholes**: Calculate option prices
4. **Fama-French**: Multi-factor risk analysis
5. **MPT**: Portfolio optimization with multiple assets

## 📊 API Endpoints

### Stock Data
- `GET /api/stock-data` - Fetch stock price data
- `GET /api/technical-indicators` - Get technical indicators
- `GET /api/support-resistance` - Calculate support/resistance levels
- `GET /api/risk-metrics` - Get comprehensive risk metrics
- `GET /api/stock-sentiment` - Analyze stock sentiment
- `GET /api/stock-report` - Generate comprehensive report

### Market Screening
- `POST /api/market-screener` - Screen stocks by criteria
- `GET /api/screening-criteria` - Get available screening options
- `GET /api/market-overview` - Get market summary statistics

### Financial Models
- `POST /api/capm` - Calculate CAPM expected return
- `POST /api/monte-carlo` - Run Monte Carlo simulation
- `POST /api/black-scholes` - Calculate option prices
- `POST /api/fama-french` - Fama-French 3-factor model
- `POST /api/mpt-optimization` - Portfolio optimization

### Portfolio & Watchlist
- `POST /api/portfolio-analysis` - Analyze portfolio performance
- `GET /api/watchlist` - Get user watchlist
- `POST /api/watchlist` - Add stock to watchlist
- `DELETE /api/watchlist` - Remove stock from watchlist

## 🎯 Key Features Explained

### Technical Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **MACD**: Trend-following momentum indicator
- **Bollinger Bands**: Volatility and trend analysis
- **Moving Averages**: Trend identification (SMA, EMA)
- **Stochastic**: Momentum oscillator for overbought/oversold
- **ATR (Average True Range)**: Volatility measurement
- **ADX (Average Directional Index)**: Trend strength
- **CCI (Commodity Channel Index)**: Cyclical trends
- **Williams %R**: Momentum oscillator
- **ROC (Rate of Change)**: Momentum measurement
- **OBV (On Balance Volume)**: Volume-price relationship
- **VWAP (Volume Weighted Average Price)**: Intraday analysis

### Risk Metrics
- **Volatility**: Annualized standard deviation of returns
- **VaR (Value at Risk)**: Maximum expected loss (95% confidence)
- **Max Drawdown**: Worst historical loss from peak
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Distribution tail thickness

### Market Screening Criteria
- **Price Range**: Minimum and maximum stock prices
- **Market Cap**: Micro, Small, Mid, Large, Mega Cap categories
- **P/E Ratio**: Price-to-earnings ratio filters
- **Dividend Yield**: Income generation potential
- **Beta**: Market correlation and volatility
- **Volume**: Trading activity levels

## 🔧 Configuration

### Markets Supported
- **NYSE (New York Stock Exchange)**: 30 major US stocks
- **PSX (Pakistan Stock Exchange)**: 20 Pakistani stocks

### Data Sources
- **Real-time Data**: yfinance API
- **Fallback Data**: Generated sample data for demonstration
- **Caching**: 5-minute cache to reduce API calls

### Customization
- **Add Markets**: Modify `MARKETS` dictionary in `app.py`
- **Add Indicators**: Extend `TECHNICAL_INDICATORS` configuration
- **Modify Screening**: Update `SCREENING_CRITERIA` settings

## 🚀 Advanced Features

### Real-time Analysis
- Live price updates
- Dynamic chart generation
- Instant technical calculations
- Real-time sentiment analysis

### Professional Tools
- Institutional-grade analytics
- Academic financial models
- Risk management tools
- Portfolio optimization

### User Experience
- Modern, responsive design
- Dark mode support
- Interactive charts
- Intuitive navigation

## 📈 Performance Optimization

- **Data Caching**: Reduces API calls and improves speed
- **Lazy Loading**: Loads data only when needed
- **Efficient Calculations**: Optimized algorithms for large datasets
- **Responsive Design**: Works on all device sizes

## 🔒 Security Features

- **Input Validation**: All user inputs are validated
- **Error Handling**: Comprehensive error management
- **Rate Limiting**: Prevents API abuse
- **Secure Headers**: HTTP security headers

## 🐛 Troubleshooting

### Common Issues
1. **TA-Lib Installation**: Use pre-compiled wheels for your platform
2. **API Rate Limits**: Built-in delays and caching handle this
3. **Data Availability**: Fallback to sample data when real data unavailable

### Performance Tips
1. **Use Caching**: Data is cached for 5 minutes
2. **Limit Date Ranges**: Shorter periods load faster
3. **Batch Operations**: Use market screener for multiple stocks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **yfinance**: For real-time stock data
- **TA-Lib**: For technical analysis functions
- **Plotly**: For interactive charts
- **Bootstrap**: For responsive UI components

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Built with ❤️ by Armaghan Tariq**

*Advanced Financial Analytics & Market Screener Platform* 