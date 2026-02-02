/* Markets page JS */

let currentMarket = 'PSX';

document.addEventListener('DOMContentLoaded', () => {
    loadMarketData('PSX');

    document.querySelectorAll('.market-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.market-btn').forEach(b => { b.classList.remove('active', 'btn-primary'); b.classList.add('btn-outline-primary'); });
            btn.classList.add('active', 'btn-primary');
            btn.classList.remove('btn-outline-primary');
            currentMarket = btn.dataset.market;
            loadMarketData(currentMarket);
        });
    });
});

async function loadMarketData(market) {
    try {
        const data = await Utils.fetchJSON(`/api/market-overview?market=${market}`);

        document.getElementById('totalStocks').textContent = data.totalStocks || '--';

        const gainersCount = data.gainers ? data.gainers.filter(s => s.changePercent > 0).length : 0;
        const losersCount = data.losers ? data.losers.filter(s => s.changePercent < 0).length : 0;
        document.getElementById('totalGainers').textContent = gainersCount;
        document.getElementById('totalLosers').textContent = losersCount;

        const sentiment = gainersCount > losersCount ? 'Bullish' : gainersCount < losersCount ? 'Bearish' : 'Neutral';
        const sentEl = document.getElementById('marketSentiment');
        sentEl.textContent = sentiment;
        sentEl.className = `stat-value ${sentiment === 'Bullish' ? 'text-success' : sentiment === 'Bearish' ? 'text-danger' : ''}`;

        // Tables
        renderMiniTable('mkGainers', data.gainers, 'change');
        renderMiniTable('mkLosers', data.losers, 'change');
        renderMiniTable('mkActive', data.mostActive, 'volume');

        // Sector chart
        loadSectorChart();
    } catch (e) {
        console.error('Market data error:', e);
    }
}

function renderMiniTable(bodyId, items, extraCol) {
    const body = document.getElementById(bodyId);
    if (!body || !items) return;

    body.innerHTML = items.map(s => {
        const cls = Utils.changeClass(s.changePercent);
        const extra = extraCol === 'volume'
            ? `<td class="text-end">${Utils.formatNumber(s.volume)}</td>`
            : `<td class="text-end ${cls}">${Utils.formatPercent(s.changePercent)}</td>`;
        return `<tr>
            <td><a href="/stocks/${s.symbol}" class="stock-link">${s.symbol}</a></td>
            <td class="text-end">${Utils.formatPrice(s.price)}</td>
            ${extra}
        </tr>`;
    }).join('');
}

async function loadSectorChart() {
    const el = document.getElementById('sectorChart');
    if (!el) return;
    try {
        const sectors = await Utils.fetchJSON('/api/sectors');
        const names = Object.keys(sectors);
        const counts = Object.values(sectors);
        const colors = ['#2563eb', '#7c3aed', '#db2777', '#ea580c', '#16a34a', '#0891b2', '#ca8a04', '#6366f1'];

        Plotly.newPlot(el, [{
            labels: names, values: counts, type: 'pie',
            hole: 0.45,
            marker: { colors: colors.slice(0, names.length) },
            textinfo: 'label+percent',
            textfont: { size: 12 },
            hovertemplate: '%{label}<br>%{value} stocks<br>%{percent}<extra></extra>',
        }], Utils.plotlyLayout({
            height: 300,
            margin: { l: 20, r: 20, t: 10, b: 10 },
            showlegend: false,
        }), Utils.plotlyConfig());
    } catch (e) { /* silent */ }
}
