// Financial Analytics Platform JavaScript

let currentTicker = 'AAPL';
let currentMarket = 'NYSE';

document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateStockOptions();
    setDefaultDates();
    
    // Check if user is already logged in
    checkLoginStatus();
    
    // If not logged in, show login form
    if (localStorage.getItem('isLoggedIn') !== 'true') {
        showLoginForm();
    }
});

function setupEventListeners() {
    document.getElementById('showRegister').addEventListener('click', showRegisterForm);
    document.getElementById('showLogin').addEventListener('click', showLoginForm);
    document.getElementById('loginFormElement').addEventListener('submit', handleLogin);
    document.getElementById('registerFormElement').addEventListener('submit', handleRegister);
    document.getElementById('logoutBtn').addEventListener('click', handleLogout);
    document.getElementById('marketSelect').addEventListener('change', updateStockOptions);
    
    // Add event listeners for calculate buttons
    setupCalculateButtons();
}

function setupCalculateButtons() {
    // CAPM Calculate Button
    const capmButton = document.querySelector('button[onclick*="calculateCAPM"]');
    if (capmButton) {
        capmButton.addEventListener('click', function(e) {
            e.preventDefault();
            calculateCAPM();
        });
    }
    
    // Monte Carlo Calculate Button
    const monteCarloButton = document.querySelector('button[onclick*="runMonteCarlo"]');
    if (monteCarloButton) {
        monteCarloButton.addEventListener('click', function(e) {
            e.preventDefault();
            runMonteCarlo();
        });
    }
    
    // Black-Scholes Calculate Button
    const blackScholesButton = document.querySelector('button[onclick*="calculateBlackScholes"]');
    if (blackScholesButton) {
        blackScholesButton.addEventListener('click', function(e) {
            e.preventDefault();
            calculateBlackScholes();
        });
    }
    
    // Fama-French Calculate Button
    const famaFrenchButton = document.querySelector('button[onclick*="calculateFamaFrench"]');
    if (famaFrenchButton) {
        famaFrenchButton.addEventListener('click', function(e) {
            e.preventDefault();
            calculateFamaFrench();
        });
    }
    
    // MPT Calculate Button
    const mptButton = document.querySelector('button[onclick*="calculateMPT"]');
    if (mptButton) {
        mptButton.addEventListener('click', function(e) {
            e.preventDefault();
            calculateMPT();
        });
    }
}

function showLoginForm() {
    document.getElementById('loginForm').classList.add('active');
    document.getElementById('registerForm').classList.remove('active');
}

function showRegisterForm() {
    document.getElementById('registerForm').classList.add('active');
    document.getElementById('loginForm').classList.remove('active');
}

function handleLogin(e) {
    e.preventDefault();
    const username = document.getElementById('loginUsername').value;
    const password = document.getElementById('loginPassword').value;
    
    if (username && password) {
        showDashboard();
        showNotification('Login successful!', 'success');
    } else {
        showNotification('Please enter valid credentials', 'error');
    }
}

function handleRegister(e) {
    e.preventDefault();
    const username = document.getElementById('registerUsername').value;
    const email = document.getElementById('registerEmail').value;
    const password = document.getElementById('registerPassword').value;
    const confirmPassword = document.getElementById('registerConfirmPassword').value;
    
    if (password !== confirmPassword) {
        showNotification('Passwords do not match', 'error');
        return;
    }
    
    if (username && email && password) {
        showDashboard();
        showNotification('Registration successful!', 'success');
    } else {
        showNotification('Please fill all fields', 'error');
    }
}

function handleLogout() {
    document.getElementById('dashboard').style.display = 'none';
    document.getElementById('authContainer').style.display = 'flex';
    localStorage.removeItem('isLoggedIn');
    showNotification('Logged out successfully', 'info');
}

function showDashboard() {
    document.getElementById('authContainer').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
    
    // Ensure dashboard stays visible
    localStorage.setItem('isLoggedIn', 'true');
    
    // Load market overview
    loadMarketOverview();
}

// Check if user is already logged in
function checkLoginStatus() {
    if (localStorage.getItem('isLoggedIn') === 'true') {
        showDashboard();
    }
}

function updateStockOptions() {
    const marketSelect = document.getElementById('marketSelect');
    const tickerSelect = document.getElementById('tickerSelect');
    const selectedMarket = marketSelect.value;
    
    tickerSelect.innerHTML = '';
    
    fetch(`/api/market-stocks/${selectedMarket}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const stocks = data.market.stocks;
                Object.keys(stocks).forEach(ticker => {
                    const option = document.createElement('option');
                    option.value = ticker;
                    option.textContent = `${ticker} - ${stocks[ticker]}`;
                    tickerSelect.appendChild(option);
                });
                
                if (Object.keys(stocks).length > 0) {
                    tickerSelect.value = Object.keys(stocks)[0];
                    currentTicker = Object.keys(stocks)[0];
                    currentMarket = selectedMarket;
                }
            }
        })
        .catch(error => {
            console.error('Error loading stocks:', error);
            showNotification('Error loading stocks', 'error');
        });
}

function setDefaultDates() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 365);  // Use 1 year of data instead of 30 days
    
    document.getElementById('endDate').value = endDate.toISOString().split('T')[0];
    document.getElementById('startDate').value = startDate.toISOString().split('T')[0];
}

async function fetchStockData() {
    showLoading(true);
    
    const ticker = document.getElementById('tickerSelect').value;
    const market = document.getElementById('marketSelect').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    const interval = document.getElementById('intervalSelect').value;
    
    currentTicker = ticker;
    currentMarket = market;
    
    try {
        const response = await fetch(`/api/stock-data?ticker=${ticker}&market=${market}&start_date=${startDate}&end_date=${endDate}&interval=${interval}`);
        const data = await response.json();
        
        if (data.success) {
            displayStockChart(data.data, data.info);
            displayStockInfo(data.info);
            if (data.cached) {
                console.log('Stock data loaded from cache');
            } else {
                console.log('Stock data loaded successfully');
            }
        } else {
            showNotification('Error loading stock data: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error loading stock data', 'error');
    } finally {
        showLoading(false);
    }
}

function displayStockChart(data, info) {
    const trace = {
        x: data.map(d => new Date(d.timestamp * 1000)),
        close: data.map(d => d.close),
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        open: data.map(d => d.open),
        decreasing: {line: {color: '#7F7F7F'}},
        increasing: {line: {color: '#17BECF'}},
        line: {color: 'rgba(31,119,180,1)'},
        fillcolor: 'rgba(31,119,180,0.3)',
        name: currentTicker,
        type: 'candlestick'
    };
    
    const layout = {
        title: `${currentTicker} Stock Price`,
        xaxis: {title: 'Date'},
        yaxis: {title: 'Price'},
        plot_bgcolor: 'rgba(0,0,0,0)',
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: {color: '#333'}
    };
    
    Plotly.newPlot('stockChart', [trace], layout);
}

function displayStockInfo(info) {
    const stockInfoDiv = document.getElementById('stockInfo');
    const changeClass = info.change >= 0 ? 'positive' : 'negative';
    const changeIcon = info.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
    
    stockInfoDiv.innerHTML = `
        <div class="info-card">
            <h6>Current Price</h6>
            <p class="price">$${info.current_price.toFixed(2)}</p>
        </div>
        <div class="info-card">
            <h6>Change</h6>
            <p class="${changeClass}">
                <i class="fas ${changeIcon}"></i>
                ${info.change.toFixed(2)}%
            </p>
        </div>
        <div class="info-card">
            <h6>Volume</h6>
            <p>${formatNumber(info.volume)}</p>
        </div>
        <div class="info-card">
            <h6>Market Cap</h6>
            <p>$${formatNumber(info.market_cap)}</p>
        </div>
    `;
}

function runMarketScreener() {
    showLoading(true);
    
    const criteria = {
        price_min: parseFloat(document.getElementById('priceMin').value) || 0,
        price_max: parseFloat(document.getElementById('priceMax').value) || 10000,
        market_cap_filter: document.getElementById('marketCapFilter').value,
        pe_min: parseFloat(document.getElementById('peMin').value) || 0,
        pe_max: parseFloat(document.getElementById('peMax').value) || 100,
        dividend_min: parseFloat(document.getElementById('dividendMin').value) || 0,
        dividend_max: parseFloat(document.getElementById('dividendMax').value) || 20,
        beta_min: parseFloat(document.getElementById('betaMin').value) || 0,
        beta_max: parseFloat(document.getElementById('betaMax').value) || 3,
        sort_by: document.getElementById('sortBy').value,
        sort_order: document.getElementById('sortOrder').value,
        limit: 50
    };
    
    fetch('/api/market-screener', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(criteria)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayScreenerResults(data.results);
            showNotification(`Found ${data.count} stocks matching criteria`, 'success');
        } else {
            showNotification('Error running screener: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error running market screener', 'error');
    })
    .finally(() => {
        showLoading(false);
    });
}

function displayScreenerResults(results) {
    const resultsDiv = document.getElementById('screenerResults');
    
    if (results.length === 0) {
        resultsDiv.innerHTML = '<p class="text-muted">No stocks found matching the criteria.</p>';
        return;
    }
    
    let html = `
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th>Price</th>
                        <th>Change %</th>
                        <th>Market Cap</th>
                        <th>P/E Ratio</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    results.forEach(stock => {
        const changeClass = stock.change_percent >= 0 ? 'text-success' : 'text-danger';
        const changeIcon = stock.change_percent >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
        
        html += `
            <tr>
                <td><strong>${stock.ticker}</strong></td>
                <td>${stock.company_name}</td>
                <td>$${stock.current_price.toFixed(2)}</td>
                <td class="${changeClass}">
                    <i class="fas ${changeIcon}"></i>
                    ${stock.change_percent.toFixed(2)}%
                </td>
                <td>$${formatNumber(stock.market_cap)}</td>
                <td>${stock.pe_ratio ? stock.pe_ratio.toFixed(2) : 'N/A'}</td>
                <td>
                    <button class="btn btn-sm btn-primary" onclick="analyzeStock('${stock.ticker}')">
                        <i class="fas fa-chart-line"></i> Analyze
                    </button>
                </td>
            </tr>
        `;
    });
    
    html += '</tbody></table></div>';
    resultsDiv.innerHTML = html;
}

function showTechnicalAnalysis() {
    if (!currentTicker) {
        showNotification('Please select a stock first', 'error');
        return;
    }
    
    showLoading(true);
    
    fetch(`/api/technical-indicators?ticker=${currentTicker}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayTechnicalIndicators(data.indicators);
                document.getElementById('technicalAnalysisSection').style.display = 'block';
                showNotification('Technical analysis loaded', 'success');
            } else {
                showNotification('Error loading technical indicators: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error loading technical analysis', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
}

function displayTechnicalIndicators(indicators) {
    const indicatorsDiv = document.getElementById('technicalIndicators');
    
    let html = '<div class="row">';
    
    // RSI
    if (indicators.RSI && indicators.RSI.length > 0) {
        const rsi = indicators.RSI[indicators.RSI.length - 1];
        const rsiClass = rsi < 30 ? 'text-success' : rsi > 70 ? 'text-danger' : 'text-warning';
        html += `
            <div class="col-md-4 mb-3">
                <div class="card">
                    <div class="card-body text-center">
                        <h6>RSI (14)</h6>
                        <h4 class="${rsiClass}">${rsi.toFixed(2)}</h4>
                        <small class="text-muted">${rsi < 30 ? 'Oversold' : rsi > 70 ? 'Overbought' : 'Neutral'}</small>
                    </div>
                </div>
            </div>
        `;
    }
    
    // MACD
    if (indicators.MACD && indicators.MACD.length > 0) {
        const macd = indicators.MACD[indicators.MACD.length - 1];
        const macdSignal = indicators.MACD_SIGNAL[indicators.MACD_SIGNAL.length - 1];
        const macdClass = macd > macdSignal ? 'text-success' : 'text-danger';
        html += `
            <div class="col-md-4 mb-3">
                <div class="card">
                    <div class="card-body text-center">
                        <h6>MACD</h6>
                        <h4 class="${macdClass}">${macd.toFixed(4)}</h4>
                        <small class="text-muted">Signal: ${macdSignal.toFixed(4)}</small>
                    </div>
                </div>
            </div>
        `;
    }
    
    html += '</div>';
    indicatorsDiv.innerHTML = html;
}

function showRiskMetrics() {
    if (!currentTicker) {
        showNotification('Please select a stock first', 'error');
        return;
    }
    
    showLoading(true);
    
    fetch(`/api/risk-metrics?ticker=${currentTicker}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayRiskMetrics(data.risk_metrics);
                document.getElementById('riskMetricsSection').style.display = 'block';
                showNotification('Risk metrics loaded', 'success');
            } else {
                showNotification('Error loading risk metrics: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error loading risk metrics', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
}

function displayRiskMetrics(metrics) {
    const metricsDiv = document.getElementById('riskMetrics');
    
    let html = '<div class="row">';
    
    const metricCards = [
        { key: 'volatility', label: 'Volatility', unit: '%', color: 'primary' },
        { key: 'var_95', label: 'VaR (95%)', unit: '%', color: 'danger' },
        { key: 'max_drawdown', label: 'Max Drawdown', unit: '%', color: 'warning' },
        { key: 'sharpe_ratio', label: 'Sharpe Ratio', unit: '', color: 'success' }
    ];
    
    metricCards.forEach(metric => {
        const value = metrics[metric.key];
        if (value !== undefined && value !== null) {
            html += `
                <div class="col-md-3 mb-3">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6>${metric.label}</h6>
                            <h4 class="text-${metric.color}">${value.toFixed(2)}${metric.unit}</h4>
                        </div>
                    </div>
                </div>
            `;
        }
    });
    
    html += '</div>';
    metricsDiv.innerHTML = html;
}

function generateStockReport() {
    if (!currentTicker) {
        showNotification('Please select a stock first', 'error');
        return;
    }
    
    showLoading(true);
    
    fetch(`/api/stock-report?ticker=${currentTicker}&market=${currentMarket}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('Stock report generated successfully', 'success');
                // Display report in a simple alert for now
                alert(`Report for ${currentTicker} generated successfully!`);
            } else {
                showNotification('Error generating report: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error generating stock report', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
}

function loadMarketOverview() {
    showLoading(true);
    
    fetch('/api/market-overview')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('Market overview loaded successfully');
            } else {
                showNotification('Error loading market overview: ' + data.error, 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error loading market overview', 'error');
        })
        .finally(() => {
            showLoading(false);
        });
}

function showMarketOverview() {
    document.getElementById('marketOverviewSection').style.display = 'block';
    document.getElementById('marketScreenerSection').style.display = 'none';
    loadMarketOverview();
}

function showMarketScreener() {
    document.getElementById('marketScreenerSection').style.display = 'block';
    document.getElementById('marketOverviewSection').style.display = 'none';
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = show ? 'flex' : 'none';
    }
    
    // Also show/hide loading text
    const loadingText = document.getElementById('loadingText');
    if (loadingText) {
        loadingText.style.display = show ? 'block' : 'none';
    }
    
    // Disable buttons during loading
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.disabled = show;
    });
}

function showNotification(message, type = 'info') {
    console.log(`${type.toUpperCase()}: ${message}`);
    // Only show alerts for errors, not for info/success
    if (type === 'error') {
        alert(`ERROR: ${message}`);
    }
}

function formatNumber(num) {
    if (num >= 1e12) {
        return (num / 1e12).toFixed(2) + 'T';
    } else if (num >= 1e9) {
        return (num / 1e9).toFixed(2) + 'B';
    } else if (num >= 1e6) {
        return (num / 1e6).toFixed(2) + 'M';
    } else if (num >= 1e3) {
        return (num / 1e3).toFixed(2) + 'K';
    } else {
        return num.toFixed(2);
    }
}

function analyzeStock(ticker) {
    currentTicker = ticker;
    document.getElementById('tickerSelect').value = ticker;
    fetchStockData();
    showTechnicalAnalysis();
    showRiskMetrics();
}

// Financial Model Functions
function calculateCAPM() {
    const beta = parseFloat(document.getElementById('beta').value);
    const marketReturn = parseFloat(document.getElementById('marketReturn').value);
    const riskFreeRate = parseFloat(document.getElementById('riskFreeRate').value);
    
    // Validate inputs
    if (isNaN(beta) || isNaN(marketReturn) || isNaN(riskFreeRate)) {
        showNotification('Please enter valid numbers for all fields', 'error');
        return;
    }
    
    showLoading(true);
    
    fetch('/api/calculate-capm', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            beta: beta,
            marketReturn: marketReturn,
            riskFreeRate: riskFreeRate
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('capmResult').innerHTML = `
                <div class="result-card">
                    <h5>Expected Return</h5>
                    <p class="price">${(data.expected_return * 100).toFixed(2)}%</p>
                    <div class="formula">
                        <small>Formula: E(Ri) = Rf + βi(Rm - Rf)</small>
                    </div>
                </div>
            `;
            console.log('CAPM calculation completed successfully');
        } else {
            showNotification('Error calculating CAPM: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error calculating CAPM', 'error');
    })
    .finally(() => {
        showLoading(false);
    });
}

function runMonteCarlo() {
    const startPrice = parseFloat(document.getElementById('startPrice').value);
    const meanReturn = parseFloat(document.getElementById('meanReturn').value);
    const volatility = parseFloat(document.getElementById('volatility').value);
    const timeHorizon = parseInt(document.getElementById('timeHorizon').value);
    const numSimulations = parseInt(document.getElementById('numSimulations').value);
    
    fetch('/api/calculate-monte-carlo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            ticker: document.getElementById('stockSelect').value || 'AAPL',
            numSimulations: numSimulations,
            timeHorizon: timeHorizon
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('monteCarloResult').innerHTML = `
                <div class="result-card">
                    <h5>Monte Carlo Simulation</h5>
                    <p><strong>Simulations:</strong> ${data.parameters.num_simulations}</p>
                    <p><strong>Time Horizon:</strong> ${data.parameters.time_horizon} days</p>
                    <p><strong>Mean Return:</strong> ${data.parameters.mean_return.toFixed(2)}%</p>
                    <p><strong>Volatility:</strong> ${data.parameters.volatility.toFixed(2)}%</p>
                </div>
            `;
            console.log('Monte Carlo simulation completed successfully');
        } else {
            showNotification('Error running Monte Carlo: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error running Monte Carlo simulation', 'error');
    });
}

function calculateBlackScholes() {
    const stockPrice = parseFloat(document.getElementById('stockPrice').value);
    const strikePrice = parseFloat(document.getElementById('strikePrice').value);
    const timeToExpiry = parseFloat(document.getElementById('timeToExpiry').value);
    const volatility = parseFloat(document.getElementById('volatilityBS').value);
    const riskFreeRate = parseFloat(document.getElementById('riskFreeRateBS').value);
    
    fetch('/api/calculate-black-scholes', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            stockPrice: stockPrice,
            strikePrice: strikePrice,
            timeToExpiry: timeToExpiry,
            volatility: volatility,
            riskFreeRate: riskFreeRate
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('blackScholesResult').innerHTML = `
                <div class="result-card">
                    <h5>Option Prices</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Call Option</h6>
                            <p class="price">$${data.call_price.toFixed(2)}</p>
                        </div>
                        <div class="col-md-6">
                            <h6>Put Option</h6>
                            <p class="price">$${data.put_price.toFixed(2)}</p>
                        </div>
                    </div>
                </div>
            `;
            console.log('Black-Scholes calculation completed successfully');
        } else {
            showNotification('Error calculating Black-Scholes: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error calculating Black-Scholes', 'error');
    });
}

function calculateFamaFrench() {
    const marketReturn = parseFloat(document.getElementById('marketReturnFF').value);
    const riskFreeRate = parseFloat(document.getElementById('riskFreeRateFF').value);
    const beta = parseFloat(document.getElementById('betaFF').value);
    const smbFactor = parseFloat(document.getElementById('smbFactor').value);
    const hmlFactor = parseFloat(document.getElementById('hmlFactor').value);
    const smbBeta = parseFloat(document.getElementById('smbBeta').value);
    const hmlBeta = parseFloat(document.getElementById('hmlBeta').value);
    
    fetch('/api/calculate-fama-french', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            ticker: document.getElementById('stockSelect').value || 'AAPL',
            riskFreeRate: riskFreeRate
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('famaFrenchResult').innerHTML = `
                <div class="result-card">
                    <h5>Fama-French 3-Factor Model</h5>
                    <p><strong>Expected Return:</strong> ${data.expected_return}%</p>
                    <div class="factors">
                        <small>Market Factor: ${data.market_contribution}%</small><br>
                        <small>SMB Factor: ${data.smb_contribution}%</small><br>
                        <small>HML Factor: ${data.hml_contribution}%</small>
                    </div>
                </div>
            `;
            console.log('Fama-French calculation completed successfully');
        } else {
            showNotification('Error calculating Fama-French: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error calculating Fama-French', 'error');
    });
}

function calculateMPT() {
    const returns = [
        parseFloat(document.getElementById('return1').value),
        parseFloat(document.getElementById('return2').value),
        parseFloat(document.getElementById('return3').value)
    ];
    
    const volatilities = [
        parseFloat(document.getElementById('volatility1').value),
        parseFloat(document.getElementById('volatility2').value),
        parseFloat(document.getElementById('volatility3').value)
    ];
    
    const correlations = [
        [1, parseFloat(document.getElementById('correlation12').value), parseFloat(document.getElementById('correlation13').value)],
        [parseFloat(document.getElementById('correlation12').value), 1, parseFloat(document.getElementById('correlation23').value)],
        [parseFloat(document.getElementById('correlation13').value), parseFloat(document.getElementById('correlation23').value), 1]
    ];
    
    const riskFreeRate = parseFloat(document.getElementById('riskFreeRateMPT').value);
    
    fetch('/api/calculate-mpt', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            tickers: ['AAPL', 'MSFT', 'GOOGL'],
            weights: [0.33, 0.33, 0.34]
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('mptResult').innerHTML = `
                <div class="result-card">
                    <h5>Modern Portfolio Theory Analysis</h5>
                    <p><strong>Expected Return:</strong> ${data.portfolio_analysis.total_return.toFixed(2)}%</p>
                    <p><strong>Portfolio Risk:</strong> ${data.portfolio_analysis.total_volatility.toFixed(2)}%</p>
                    <p><strong>Sharpe Ratio:</strong> ${data.portfolio_analysis.sharpe_ratio.toFixed(3)}</p>
                    <p><strong>Diversification Score:</strong> ${(data.portfolio_analysis.diversification_score * 100).toFixed(1)}%</p>
                </div>
            `;
            console.log('MPT calculation completed successfully');
        } else {
            showNotification('Error calculating MPT: ' + data.error, 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error calculating MPT', 'error');
    });
} 