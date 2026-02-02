/* ============================================
   FINBRIDGE - Main JavaScript
   Global: Search, Ticker Strip, Utilities
   PSX-Only
   ============================================ */

const Utils = {
    formatNumber(n) {
        if (n === null || n === undefined) return '--';
        if (Math.abs(n) >= 1e12) return (n / 1e12).toFixed(2) + 'T';
        if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + 'B';
        if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
        if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
        return Number(n).toLocaleString();
    },
    formatPrice(n) {
        if (n === null || n === undefined) return '--';
        return 'PKR ' + Number(n).toFixed(2);
    },
    formatPercent(n) {
        if (n === null || n === undefined) return '--';
        const sign = n >= 0 ? '+' : '';
        return sign + Number(n).toFixed(2) + '%';
    },
    changeClass(n) {
        if (n > 0) return 'text-success';
        if (n < 0) return 'text-danger';
        return '';
    },
    changeArrow(n) {
        if (n > 0) return '<i class="fas fa-caret-up me-1"></i>';
        if (n < 0) return '<i class="fas fa-caret-down me-1"></i>';
        return '';
    },
    async fetchJSON(url, options) {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
    },
    plotlyLayout(extra = {}) {
        return Object.assign({
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { family: 'Inter, sans-serif', color: '#6b7280', size: 12 },
            margin: { l: 50, r: 20, t: 10, b: 40 },
            xaxis: { gridcolor: '#f0f1f3', linecolor: '#e5e7eb', zerolinecolor: '#e5e7eb' },
            yaxis: { gridcolor: '#f0f1f3', linecolor: '#e5e7eb', zerolinecolor: '#e5e7eb' },
            showlegend: false,
            hovermode: 'x unified',
        }, extra);
    },
    plotlyConfig() {
        return { responsive: true, displayModeBar: false };
    }
};

window.Utils = Utils;

// ---------- DOMContentLoaded ----------
document.addEventListener('DOMContentLoaded', () => {
    // Search
    initSearch('globalSearch', 'searchResults');

    // Keyboard shortcut /
    document.addEventListener('keydown', (e) => {
        if (e.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) {
            e.preventDefault();
            const el = document.getElementById('globalSearch');
            if (el) el.focus();
        }
    });

    // Ticker strip (PSX)
    loadTickerStrip();

    // Watchlist sidebar
    const openWl = document.getElementById('openWatchlist');
    if (openWl) {
        openWl.addEventListener('click', (e) => {
            e.preventDefault();
            const sidebar = new bootstrap.Offcanvas(document.getElementById('watchlistSidebar'));
            sidebar.show();
            loadWatchlist();
        });
    }
});

// ---------- Search ----------
function initSearch(inputId, resultsId) {
    const input = document.getElementById(inputId);
    const results = document.getElementById(resultsId);
    if (!input || !results) return;

    let timer = null;
    input.addEventListener('input', () => {
        clearTimeout(timer);
        const q = input.value.trim();
        if (q.length < 1) { results.classList.remove('show'); return; }
        timer = setTimeout(async () => {
            try {
                const data = await Utils.fetchJSON(`/api/search?q=${encodeURIComponent(q)}`);
                if (!data.length) {
                    results.innerHTML = '<div class="p-3 text-muted text-center">No PSX stocks found</div>';
                } else {
                    results.innerHTML = data.map(s => `
                        <a href="/stocks/${s.symbol}" class="search-result-item">
                            <div>
                                <span class="symbol">${s.symbol}</span>
                                <span class="name ms-2">${s.name}</span>
                            </div>
                            <span class="market">PSX</span>
                        </a>
                    `).join('');
                }
                results.classList.add('show');
            } catch (err) {
                results.innerHTML = '<div class="p-3 text-muted text-center">Search error</div>';
                results.classList.add('show');
            }
        }, 200);
    });

    input.addEventListener('blur', () => setTimeout(() => results.classList.remove('show'), 200));
    input.addEventListener('focus', () => { if (input.value.trim().length > 0) results.classList.add('show'); });
}

// ---------- Ticker Strip (PSX only) ----------
async function loadTickerStrip() {
    const container = document.getElementById('tickerTrack');
    if (!container) return;
    try {
        const data = await Utils.fetchJSON('/api/market-overview?market=PSX');
        const all = [...(data.gainers || []).slice(0, 5), ...(data.losers || []).slice(0, 5), ...(data.mostActive || []).slice(0, 5)];
        const seen = new Set();
        const items = all.filter(s => { if (seen.has(s.symbol)) return false; seen.add(s.symbol); return true; });

        const html = items.map(s => {
            const cls = s.changePercent >= 0 ? 'up' : 'down';
            return `<span class="ticker-item">
                <span class="sym">${s.symbol}</span>
                <span class="prc">${Utils.formatPrice(s.price)}</span>
                <span class="${cls}">${Utils.formatPercent(s.changePercent)}</span>
            </span>`;
        }).join('');
        container.innerHTML = html + html; // duplicate for infinite scroll
    } catch (err) {
        container.innerHTML = '<span class="ticker-item"><span class="text-muted">PSX data loading...</span></span>';
    }
}

// ---------- Watchlist ----------
async function loadWatchlist() {
    const body = document.getElementById('watchlistBody');
    if (!body) return;
    try {
        const data = await Utils.fetchJSON('/api/watchlist');
        if (!data.length) {
            body.innerHTML = '<p class="text-muted text-center mt-4">Your watchlist is empty. Add stocks from their detail pages.</p>';
            return;
        }
        body.innerHTML = data.map(s => `
            <div class="d-flex justify-content-between align-items-center p-2 border-bottom">
                <div>
                    <a href="/stocks/${s.symbol}" class="fw-bold">${s.symbol}</a>
                    <div class="small text-muted">${s.name}</div>
                </div>
                <div class="text-end">
                    <div class="fw-semibold">${Utils.formatPrice(s.price)}</div>
                    <div class="small ${Utils.changeClass(s.changePercent)}">${Utils.formatPercent(s.changePercent)}</div>
                </div>
            </div>
        `).join('');
    } catch (err) {
        body.innerHTML = '<p class="text-muted text-center mt-4">Sign in to use your watchlist.</p>';
    }
}

// ---------- Reusable Stock Row ----------
function stockRow(s, rank) {
    const cls = Utils.changeClass(s.changePercent);
    return `<tr>
        ${rank !== undefined ? `<td>${rank}</td>` : ''}
        <td><a href="/stocks/${s.symbol}" class="stock-link">${s.symbol}</a></td>
        <td>${s.name || ''}</td>
        <td class="text-end">${Utils.formatPrice(s.price)}</td>
        <td class="text-end ${cls}">${Utils.changeArrow(s.change)}${s.change >= 0 ? '+' : ''}${Number(s.change).toFixed(2)}</td>
        <td class="text-end ${cls}">${Utils.formatPercent(s.changePercent)}</td>
        <td class="text-end">${Utils.formatNumber(s.volume)}</td>
        ${s.marketCap !== undefined ? `<td class="text-end">${Utils.formatNumber(s.marketCap)}</td>` : ''}
    </tr>`;
}

window.stockRow = stockRow;
