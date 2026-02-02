/* Stock Detail page JS */

let chartPeriod = '1y';
let chartType = 'candlestick';

document.addEventListener('DOMContentLoaded', () => {
    loadStockQuote();
    loadOverviewChart('1y');
    loadKeyStats();
    loadSentiment();

    // Overview period buttons
    document.querySelectorAll('#overviewPeriod button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#overviewPeriod button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadOverviewChart(btn.dataset.period);
        });
    });

    // Chart period buttons
    document.querySelectorAll('#chartPeriod button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#chartPeriod button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            chartPeriod = btn.dataset.period;
            loadMainChart();
        });
    });

    // Chart type buttons
    document.querySelectorAll('#chartType button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#chartType button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            chartType = btn.dataset.type;
            loadMainChart();
        });
    });

    // Tab events - load data on tab switch
    document.querySelectorAll('#stockTabs button').forEach(btn => {
        btn.addEventListener('shown.bs.tab', (e) => {
            const target = e.target.getAttribute('data-bs-target');
            if (target === '#chartTab') loadMainChart();
            if (target === '#financialsTab') loadFinancials();
            if (target === '#techTab') loadTechnicals();
            if (target === '#riskTab') loadRisk();
        });
    });

    // Financial models
    document.getElementById('calcCapm')?.addEventListener('click', calcCapm);
    document.getElementById('calcBs')?.addEventListener('click', calcBlackScholes);
    document.getElementById('calcMc')?.addEventListener('click', calcMonteCarlo);

    // Watchlist
    document.getElementById('addWatchlistBtn')?.addEventListener('click', addToWatchlist);
});

// ---------- Stock Quote ----------
async function loadStockQuote() {
    try {
        const q = await Utils.fetchJSON(`/api/stock-quote?ticker=${TICKER}`);
        document.getElementById('stockName').innerHTML = `${q.name} <small class="text-muted">(${q.symbol})</small>`;
        document.getElementById('stockExchange').textContent = q.exchange;
        document.getElementById('stockSector').textContent = q.sector || '--';
        document.getElementById('stockPrice').textContent = Utils.formatPrice(q.price);

        const cls = Utils.changeClass(q.changePercent);
        document.getElementById('stockChange').innerHTML = `<span class="${cls}">${Utils.changeArrow(q.change)}${q.change >= 0 ? '+' : ''}${Number(q.change).toFixed(2)}</span>`;
        document.getElementById('stockChangePct').innerHTML = `<span class="${cls}">(${Utils.formatPercent(q.changePercent)})</span>`;
    } catch (e) { /* silent */ }
}

// ---------- Overview Chart ----------
async function loadOverviewChart(period) {
    const el = document.getElementById('overviewChart');
    if (!el) return;
    try {
        const data = await Utils.fetchJSON(`/api/stock-data?ticker=${TICKER}&period=${period}&interval=1d`);
        if (!data.data) return;

        const dates = data.data.map(d => d.date);
        const closes = data.data.map(d => d.close);
        const first = closes[0] || 0;
        const last = closes[closes.length - 1] || 0;
        const color = last >= first ? '#16a34a' : '#dc2626';
        const fillColor = last >= first ? 'rgba(22,163,74,0.08)' : 'rgba(220,38,38,0.08)';

        Plotly.newPlot(el, [{
            x: dates, y: closes,
            type: 'scatter', mode: 'lines',
            fill: 'tozeroy', fillcolor: fillColor,
            line: { color, width: 2 },
            hovertemplate: '%{x}<br>PKR %{y:.2f}<extra></extra>'
        }], Utils.plotlyLayout({ height: 400 }), Utils.plotlyConfig());
    } catch (e) { el.innerHTML = '<p class="text-muted text-center p-4">Chart unavailable</p>'; }
}

// ---------- Main Chart ----------
async function loadMainChart() {
    const el = document.getElementById('mainChart');
    const volEl = document.getElementById('volumeChart');
    if (!el) return;

    try {
        const data = await Utils.fetchJSON(`/api/stock-data?ticker=${TICKER}&period=${chartPeriod}&interval=1d`);
        if (!data.data) return;

        const dates = data.data.map(d => d.date);
        const opens = data.data.map(d => d.open);
        const highs = data.data.map(d => d.high);
        const lows = data.data.map(d => d.low);
        const closes = data.data.map(d => d.close);
        const volumes = data.data.map(d => d.volume);

        let trace;
        if (chartType === 'candlestick') {
            trace = {
                x: dates, open: opens, high: highs, low: lows, close: closes,
                type: 'candlestick',
                increasing: { line: { color: '#16a34a' }, fillcolor: '#16a34a' },
                decreasing: { line: { color: '#dc2626' }, fillcolor: '#dc2626' },
            };
        } else if (chartType === 'line') {
            trace = {
                x: dates, y: closes, type: 'scatter', mode: 'lines',
                line: { color: '#2563eb', width: 2 },
                hovertemplate: '%{x}<br>PKR %{y:.2f}<extra></extra>'
            };
        } else {
            trace = {
                x: dates, y: closes, type: 'scatter', mode: 'lines',
                fill: 'tozeroy', fillcolor: 'rgba(37,99,235,0.1)',
                line: { color: '#2563eb', width: 2 },
                hovertemplate: '%{x}<br>PKR %{y:.2f}<extra></extra>'
            };
        }

        Plotly.newPlot(el, [trace], Utils.plotlyLayout({
            height: 550,
            xaxis: { rangeslider: { visible: false }, gridcolor: 'transparent' },
        }), Utils.plotlyConfig());

        // Volume
        if (volEl) {
            const colors = closes.map((c, i) => i > 0 && c >= closes[i - 1] ? '#16a34a' : '#dc2626');
            Plotly.newPlot(volEl, [{
                x: dates, y: volumes, type: 'bar',
                marker: { color: colors, opacity: 0.6 },
                hovertemplate: '%{x}<br>Vol: %{y:,.0f}<extra></extra>'
            }], Utils.plotlyLayout({
                height: 150,
                margin: { l: 50, r: 20, t: 0, b: 30 },
                yaxis: { gridcolor: 'transparent' },
            }), Utils.plotlyConfig());
        }
    } catch (e) { el.innerHTML = '<p class="text-muted text-center p-4">Chart unavailable</p>'; }
}

// ---------- Key Stats ----------
async function loadKeyStats() {
    const el = document.getElementById('keyStats');
    if (!el) return;
    try {
        const q = await Utils.fetchJSON(`/api/stock-quote?ticker=${TICKER}`);
        const rows = [
            ['Previous Close', Utils.formatPrice(q.prevClose)],
            ['Open', Utils.formatPrice(q.open)],
            ['Day High', Utils.formatPrice(q.high)],
            ['Day Low', Utils.formatPrice(q.low)],
            ['52-Week High', Utils.formatPrice(q.week52High)],
            ['52-Week Low', Utils.formatPrice(q.week52Low)],
            ['Volume', Utils.formatNumber(q.volume)],
            ['Avg Volume', Utils.formatNumber(q.avgVolume)],
            ['Market Cap', Utils.formatNumber(q.marketCap)],
            ['P/E Ratio', q.pe ? Number(q.pe).toFixed(2) : '--'],
            ['EPS', q.eps ? 'PKR ' + Number(q.eps).toFixed(2) : '--'],
            ['Beta', q.beta ? Number(q.beta).toFixed(2) : '--'],
            ['Dividend Yield', q.dividend ? (q.dividend * 100).toFixed(2) + '%' : '--'],
        ];
        el.innerHTML = rows.map(([label, val]) => `<tr><td>${label}</td><td>${val}</td></tr>`).join('');
    } catch (e) { el.innerHTML = '<tr><td colspan="2" class="text-muted">Unable to load</td></tr>'; }
}

// ---------- Sentiment ----------
async function loadSentiment() {
    const el = document.getElementById('sentimentCard');
    if (!el) return;
    try {
        const s = await Utils.fetchJSON(`/api/sentiment?ticker=${TICKER}`);
        const cls = s.signal === 'Buy' ? 'sentiment-buy' : s.signal === 'Sell' ? 'sentiment-sell' : 'sentiment-hold';
        el.innerHTML = `
            <div class="sentiment-badge ${cls} mb-3">${s.signal}</div>
            <div class="mb-2"><strong>Score:</strong> ${s.score}/100</div>
            <div class="progress mb-3" style="height:8px;">
                <div class="progress-bar ${s.score >= 50 ? 'bg-success' : 'bg-danger'}" style="width:${s.score}%"></div>
            </div>
            <div class="small text-muted">
                <div>RSI: ${s.rsi}</div>
                <div>MACD: ${s.macdSignal}</div>
                <div>Trend: ${s.trendSignal}</div>
            </div>
        `;
    } catch (e) { el.innerHTML = '<p class="text-muted">Unable to load sentiment</p>'; }
}

// ---------- Financials ----------
async function loadFinancials() {
    try {
        const fin = await Utils.fetchJSON(`/api/financials?ticker=${TICKER}`);

        // Overview
        const ov = fin.overview;
        const ovEl = document.getElementById('financialOverview');
        if (ovEl) {
            const items = [
                ['Market Cap', Utils.formatNumber(ov.marketCap)],
                ['Revenue', Utils.formatNumber(ov.revenue)],
                ['Revenue Growth', (ov.revenueGrowth * 100).toFixed(1) + '%'],
                ['Earnings Growth', (ov.earningsGrowth * 100).toFixed(1) + '%'],
                ['Profit Margin', (ov.profitMargin * 100).toFixed(1) + '%'],
                ['ROE', (ov.roe * 100).toFixed(1) + '%'],
                ['Debt/Equity', Number(ov.debtToEquity).toFixed(1)],
                ['Free Cash Flow', Utils.formatNumber(ov.freeCashflow)],
                ['EPS', 'PKR ' + Number(ov.eps).toFixed(2)],
                ['P/E', Number(ov.pe).toFixed(2)],
                ['Forward P/E', Number(ov.forwardPE).toFixed(2)],
                ['P/B', Number(ov.priceToBook).toFixed(2)],
            ];
            ovEl.innerHTML = items.map(([l, v]) => `
                <div class="col-md-3 col-sm-6">
                    <div class="p-3 border rounded">
                        <div class="small text-muted">${l}</div>
                        <div class="fw-bold">${v}</div>
                    </div>
                </div>
            `).join('');
        }

        // Income Statement
        buildFinancialTable('incomeHead', 'incomeBody', fin.incomeStatement, [
            ['Revenue', 'revenue'], ['Cost of Revenue', 'costOfRevenue'],
            ['Gross Profit', 'grossProfit'], ['Operating Expenses', 'operatingExpenses'],
            ['Operating Income', 'operatingIncome'], ['Net Income', 'netIncome'],
        ]);

        // Balance Sheet
        buildFinancialTable('balanceHead', 'balanceBody', fin.balanceSheet, [
            ['Total Assets', 'totalAssets'], ['Total Liabilities', 'totalLiabilities'],
            ['Stockholders Equity', 'stockholdersEquity'], ['Cash', 'cash'],
            ['Total Debt', 'totalDebt'],
        ]);
    } catch (e) { console.error('Financials error:', e); }
}

function buildFinancialTable(headId, bodyId, data, fields) {
    const head = document.getElementById(headId);
    const body = document.getElementById(bodyId);
    if (!head || !body || !data || !data.length) return;

    head.innerHTML = '<th>Metric</th>' + data.map(d => `<th class="text-end">${d.year}</th>`).join('');
    body.innerHTML = fields.map(([label, key]) => {
        return '<tr><td>' + label + '</td>' + data.map(d => `<td class="text-end">${Utils.formatNumber(d[key])}</td>`).join('') + '</tr>';
    }).join('');
}

// ---------- Technical Indicators ----------
async function loadTechnicals() {
    try {
        const t = await Utils.fetchJSON(`/api/technical-indicators?ticker=${TICKER}&period=1y&interval=1d`);

        // Price + MA chart
        const techEl = document.getElementById('techChart');
        if (techEl) {
            const traces = [
                { x: t.dates, y: t.close, name: 'Close', line: { color: '#2563eb', width: 2 } },
                { x: t.dates, y: t.sma20, name: 'SMA 20', line: { color: '#f59e0b', width: 1, dash: 'dot' } },
                { x: t.dates, y: t.sma50, name: 'SMA 50', line: { color: '#8b5cf6', width: 1, dash: 'dot' } },
                { x: t.dates, y: t.bb_upper, name: 'BB Upper', line: { color: '#94a3b8', width: 1 } },
                { x: t.dates, y: t.bb_lower, name: 'BB Lower', line: { color: '#94a3b8', width: 1 }, fill: 'tonexty', fillcolor: 'rgba(148,163,184,0.08)' },
            ].map(tr => Object.assign({ type: 'scatter', mode: 'lines' }, tr));

            Plotly.newPlot(techEl, traces, Utils.plotlyLayout({ height: 450, showlegend: true, legend: { orientation: 'h', y: 1.1 } }), Utils.plotlyConfig());
        }

        // RSI
        const rsiEl = document.getElementById('rsiChart');
        if (rsiEl) {
            Plotly.newPlot(rsiEl, [
                { x: t.dates, y: t.rsi, type: 'scatter', mode: 'lines', line: { color: '#8b5cf6', width: 1.5 }, name: 'RSI' },
                { x: t.dates, y: t.dates.map(() => 70), type: 'scatter', mode: 'lines', line: { color: '#dc2626', width: 1, dash: 'dash' }, name: 'Overbought' },
                { x: t.dates, y: t.dates.map(() => 30), type: 'scatter', mode: 'lines', line: { color: '#16a34a', width: 1, dash: 'dash' }, name: 'Oversold' },
            ], Utils.plotlyLayout({
                height: 200, showlegend: true, legend: { orientation: 'h', y: 1.15 },
                yaxis: { range: [0, 100], gridcolor: 'transparent' },
            }), Utils.plotlyConfig());
        }

        // MACD
        const macdEl = document.getElementById('macdChart');
        if (macdEl) {
            const colors = t.macd_histogram.map(v => v >= 0 ? '#16a34a' : '#dc2626');
            Plotly.newPlot(macdEl, [
                { x: t.dates, y: t.macd, type: 'scatter', mode: 'lines', line: { color: '#2563eb', width: 1.5 }, name: 'MACD' },
                { x: t.dates, y: t.macd_signal, type: 'scatter', mode: 'lines', line: { color: '#f59e0b', width: 1.5 }, name: 'Signal' },
                { x: t.dates, y: t.macd_histogram, type: 'bar', marker: { color: colors, opacity: 0.6 }, name: 'Histogram' },
            ], Utils.plotlyLayout({
                height: 200, showlegend: true, legend: { orientation: 'h', y: 1.15 },
            }), Utils.plotlyConfig());
        }

        // Indicator values table
        const indEl = document.getElementById('techIndicators');
        if (indEl) {
            const last = (arr) => { for (let i = arr.length - 1; i >= 0; i--) if (arr[i] !== null) return arr[i]; return null; };
            const rows = [
                ['RSI (14)', last(t.rsi) ? Number(last(t.rsi)).toFixed(2) : '--'],
                ['SMA 20', last(t.sma20) ? 'PKR ' + Number(last(t.sma20)).toFixed(2) : '--'],
                ['SMA 50', last(t.sma50) ? 'PKR ' + Number(last(t.sma50)).toFixed(2) : '--'],
                ['SMA 200', last(t.sma200) ? 'PKR ' + Number(last(t.sma200)).toFixed(2) : '--'],
                ['EMA 12', last(t.ema12) ? 'PKR ' + Number(last(t.ema12)).toFixed(2) : '--'],
                ['EMA 26', last(t.ema26) ? 'PKR ' + Number(last(t.ema26)).toFixed(2) : '--'],
                ['MACD', last(t.macd) ? Number(last(t.macd)).toFixed(4) : '--'],
                ['MACD Signal', last(t.macd_signal) ? Number(last(t.macd_signal)).toFixed(4) : '--'],
                ['BB Upper', last(t.bb_upper) ? 'PKR ' + Number(last(t.bb_upper)).toFixed(2) : '--'],
                ['BB Lower', last(t.bb_lower) ? 'PKR ' + Number(last(t.bb_lower)).toFixed(2) : '--'],
                ['ATR', last(t.atr) ? Number(last(t.atr)).toFixed(4) : '--'],
                ['Stoch %K', last(t.stoch_k) ? Number(last(t.stoch_k)).toFixed(2) : '--'],
                ['Stoch %D', last(t.stoch_d) ? Number(last(t.stoch_d)).toFixed(2) : '--'],
                ['OBV', last(t.obv) ? Utils.formatNumber(last(t.obv)) : '--'],
                ['VWAP', last(t.vwap) ? 'PKR ' + Number(last(t.vwap)).toFixed(2) : '--'],
            ];
            indEl.innerHTML = rows.map(([l, v]) => `<tr><td>${l}</td><td>${v}</td></tr>`).join('');
        }
    } catch (e) { console.error('Technicals error:', e); }
}

// ---------- Risk Metrics ----------
async function loadRisk() {
    const el = document.getElementById('riskContent');
    if (!el) return;
    try {
        const r = await Utils.fetchJSON(`/api/risk-metrics?ticker=${TICKER}`);
        const items = [
            ['Annual Return', r.annualizedReturn + '%', r.annualizedReturn >= 0 ? 'text-success' : 'text-danger'],
            ['Volatility', r.annualizedVolatility + '%', ''],
            ['Sharpe Ratio', r.sharpeRatio, r.sharpeRatio >= 1 ? 'text-success' : ''],
            ['Sortino Ratio', r.sortinoRatio, ''],
            ['Calmar Ratio', r.calmarRatio, ''],
            ['Max Drawdown', r.maxDrawdown + '%', 'text-danger'],
            ['VaR 95%', r.var95 + '%', 'text-danger'],
            ['VaR 99%', r.var99 + '%', 'text-danger'],
            ['Beta', r.beta, ''],
            ['Correlation', r.correlation, ''],
            ['Skewness', r.skewness, ''],
            ['Kurtosis', r.kurtosis, ''],
        ];
        el.innerHTML = items.map(([l, v, cls]) => `
            <div class="col-md-3 col-sm-6">
                <div class="card risk-card">
                    <div class="metric-value ${cls}">${v}</div>
                    <div class="metric-label">${l}</div>
                </div>
            </div>
        `).join('');
    } catch (e) { el.innerHTML = '<div class="col-12 text-muted text-center">Unable to load risk metrics</div>'; }
}

// ---------- Financial Models ----------
async function calcCapm() {
    const rf = document.getElementById('capmRf').value;
    const el = document.getElementById('capmResult');
    try {
        const r = await Utils.fetchJSON('/api/capm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: TICKER, risk_free_rate: rf }),
        });
        el.innerHTML = `<div class="model-result">
            <div class="result-row"><span class="result-label">Beta</span><span class="result-value">${r.beta}</span></div>
            <div class="result-row"><span class="result-label">Risk-Free Rate</span><span class="result-value">${r.riskFreeRate}%</span></div>
            <div class="result-row"><span class="result-label">Market Return</span><span class="result-value">${r.marketReturn}%</span></div>
            <div class="result-row"><span class="result-label">Risk Premium</span><span class="result-value">${r.riskPremium}%</span></div>
            <div class="result-row"><span class="result-label"><strong>Expected Return</strong></span><span class="result-value text-primary"><strong>${r.expectedReturn}%</strong></span></div>
        </div>`;
    } catch (e) { el.innerHTML = '<p class="text-danger">Calculation failed</p>'; }
}

async function calcBlackScholes() {
    const el = document.getElementById('bsResult');
    try {
        const r = await Utils.fetchJSON('/api/black-scholes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: TICKER,
                strike_price: document.getElementById('bsStrike').value,
                expiry_days: document.getElementById('bsExpiry').value,
                risk_free_rate: document.getElementById('bsRf').value,
                volatility: document.getElementById('bsVol').value || undefined,
            }),
        });
        el.innerHTML = `<div class="model-result">
            <div class="result-row"><span class="result-label">Spot Price</span><span class="result-value">PKR ${r.spotPrice}</span></div>
            <div class="result-row"><span class="result-label">Volatility</span><span class="result-value">${r.volatility}%</span></div>
            <div class="result-row"><span class="result-label"><strong>Call Price</strong></span><span class="result-value text-success"><strong>PKR ${r.callPrice}</strong></span></div>
            <div class="result-row"><span class="result-label"><strong>Put Price</strong></span><span class="result-value text-danger"><strong>PKR ${r.putPrice}</strong></span></div>
            <div class="result-row"><span class="result-label">Delta (Call/Put)</span><span class="result-value">${r.callDelta} / ${r.putDelta}</span></div>
            <div class="result-row"><span class="result-label">Gamma</span><span class="result-value">${r.gamma}</span></div>
            <div class="result-row"><span class="result-label">Theta</span><span class="result-value">${r.theta}</span></div>
            <div class="result-row"><span class="result-label">Vega</span><span class="result-value">${r.vega}</span></div>
        </div>`;
    } catch (e) { el.innerHTML = '<p class="text-danger">Calculation failed</p>'; }
}

async function calcMonteCarlo() {
    const chartEl = document.getElementById('mcChart');
    const resEl = document.getElementById('mcResult');
    try {
        const r = await Utils.fetchJSON('/api/monte-carlo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: TICKER,
                simulations: document.getElementById('mcSims').value,
                days: document.getElementById('mcDays').value,
            }),
        });

        // Chart
        const traces = r.paths.map((path, i) => ({
            y: path, type: 'scatter', mode: 'lines',
            line: { color: `hsl(${(i * 7) % 360}, 60%, 55%)`, width: 0.5 },
            opacity: 0.4, showlegend: false,
            hoverinfo: 'skip',
        }));
        traces.push({
            y: Array(r.days).fill(r.currentPrice),
            type: 'scatter', mode: 'lines',
            line: { color: '#fff', width: 2, dash: 'dash' },
            name: 'Current',
        });

        Plotly.newPlot(chartEl, traces, Utils.plotlyLayout({
            height: 400, showlegend: true,
            yaxis: { title: 'Price (PKR)' },
            xaxis: { title: 'Days' },
        }), Utils.plotlyConfig());

        resEl.innerHTML = `<div class="model-result">
            <div class="result-row"><span class="result-label">Current Price</span><span class="result-value">PKR ${r.currentPrice}</span></div>
            <div class="result-row"><span class="result-label">Mean Forecast</span><span class="result-value">PKR ${r.meanPrice}</span></div>
            <div class="result-row"><span class="result-label">Median Forecast</span><span class="result-value">PKR ${r.medianPrice}</span></div>
            <div class="result-row"><span class="result-label">5th Percentile (Bear)</span><span class="result-value text-danger">PKR ${r.percentile5}</span></div>
            <div class="result-row"><span class="result-label">95th Percentile (Bull)</span><span class="result-value text-success">PKR ${r.percentile95}</span></div>
            <div class="result-row"><span class="result-label">Std Deviation</span><span class="result-value">PKR ${r.stdDev}</span></div>
        </div>`;
    } catch (e) {
        resEl.innerHTML = '<p class="text-danger">Simulation failed</p>';
    }
}

// ---------- Watchlist ----------
async function addToWatchlist() {
    const btn = document.getElementById('addWatchlistBtn');
    try {
        await Utils.fetchJSON('/api/watchlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: TICKER }),
        });
        btn.innerHTML = '<i class="fas fa-star text-warning me-1"></i>In Watchlist';
        btn.classList.remove('btn-outline-primary');
        btn.classList.add('btn-outline-warning');
    } catch (e) {
        // User not logged in
        window.location.href = '/auth/login';
    }
}
