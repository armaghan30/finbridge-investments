/* Screener page JS - with 30 Indicator Scoring System */

let currentSort = { by: 'marketCap', dir: 'desc' };
let currentPage = 1;
let allResults = []; // store for modal lookups

const CATEGORY_ORDER = [
    'Basic', 'Performance', 'Valuation', 'Risk Indicators',
    'Short-Term Liquidity', 'Leverage', 'Efficiency', 'Growth', 'Volatility'
];

const CATEGORY_ICONS = {
    'Basic': 'fas fa-building',
    'Performance': 'fas fa-chart-line',
    'Valuation': 'fas fa-tags',
    'Risk Indicators': 'fas fa-shield-alt',
    'Short-Term Liquidity': 'fas fa-tint',
    'Leverage': 'fas fa-balance-scale',
    'Efficiency': 'fas fa-cogs',
    'Growth': 'fas fa-rocket',
    'Volatility': 'fas fa-bolt'
};

document.addEventListener('DOMContentLoaded', () => {
    loadScreener();

    document.getElementById('applyFilters').addEventListener('click', () => { currentPage = 1; loadScreener(); });
    document.getElementById('resetFilters').addEventListener('click', resetFilters);

    // Sortable headers
    document.querySelectorAll('.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (currentSort.by === col) {
                currentSort.dir = currentSort.dir === 'desc' ? 'asc' : 'desc';
            } else {
                currentSort = { by: col, dir: 'desc' };
            }
            loadScreener();
        });
    });
});

function getFilters() {
    const f = {
        market: document.getElementById('filterMarket') ? document.getElementById('filterMarket').value : 'PSX',
        sort_by: currentSort.by,
        sort_dir: currentSort.dir,
        page: currentPage,
        per_page: 25,
        include_indicators: true,
    };

    const add = (id, key) => { const el = document.getElementById(id); if (el && el.value) f[key] = el.value; };
    add('filterSector', 'sector');
    add('filterMarketCap', 'market_cap');
    add('filterPriceMin', 'price_min');
    add('filterPriceMax', 'price_max');
    add('filterPeMin', 'pe_min');
    add('filterPeMax', 'pe_max');
    add('filterDivMin', 'div_min');
    add('filterDivMax', 'div_max');
    add('filterBetaMin', 'beta_min');
    add('filterBetaMax', 'beta_max');
    add('filterRisk', 'risk_level');

    return f;
}

function scoreClass(score) {
    if (score === null || score === undefined) return '';
    if (score <= 1.5) return 'score-1';
    if (score <= 2.5) return 'score-2';
    if (score <= 3.25) return 'score-3';
    return 'score-4';
}

function scoreBadgeHTML(score, label) {
    if (score === null || score === undefined) return '<span class="text-muted">--</span>';
    const cls = scoreClass(score);
    return `<span class="score-badge ${cls}" title="${label || ''}">${score.toFixed(1)}</span>`;
}

function scoreLabel(score) {
    if (score === null || score === undefined) return 'N/A';
    if (score <= 1.5) return 'Low Risk';
    if (score <= 2.5) return 'Medium Risk';
    if (score <= 3.25) return 'Med-High Risk';
    return 'High Risk';
}

async function loadScreener() {
    const body = document.getElementById('screenerBody');
    body.innerHTML = '<tr><td colspan="11" class="text-center py-5"><div class="spinner-border spinner-border-sm"></div> Screening & scoring stocks...</td></tr>';

    try {
        const filters = getFilters();
        const res = await Utils.fetchJSON('/api/screener', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(filters),
        });

        if (!res.results || !res.results.length) {
            body.innerHTML = '<tr><td colspan="11" class="text-center py-5 text-muted">No stocks match your criteria. Try adjusting filters.</td></tr>';
            document.getElementById('resultCount').textContent = '0 results';
            document.getElementById('paginationInfo').textContent = '';
            document.getElementById('pagination').innerHTML = '';
            return;
        }

        // Client-side risk level filter
        let results = res.results;
        const riskFilter = filters.risk_level;
        if (riskFilter) {
            const ranges = { low: [0, 1.5], medium: [1.5, 2.5], medhigh: [2.5, 3.25], high: [3.25, 5] };
            const [lo, hi] = ranges[riskFilter] || [0, 5];
            results = results.filter(s => s.overallScore >= lo && s.overallScore < hi);
        }

        allResults = results;

        body.innerHTML = results.map((s, idx) => `
            <tr class="screener-row" data-idx="${idx}">
                <td><a href="/stocks/${s.symbol}" class="stock-link">${s.symbol}</a></td>
                <td>${s.name}</td>
                <td class="text-end">${Utils.formatPrice(s.price)}</td>
                <td class="text-end ${Utils.changeClass(s.changePercent)}">${Utils.changeArrow(s.changePercent)}${Utils.formatPercent(s.changePercent)}</td>
                <td class="text-end">${Utils.formatNumber(s.marketCap)}</td>
                <td class="text-end">${s.pe || '--'}</td>
                <td class="text-end">${s.dividendYield ? s.dividendYield + '%' : '--'}</td>
                <td class="text-end">${s.beta || '--'}</td>
                <td class="text-center">${scoreBadgeHTML(s.overallScore, s.scoreLabel)}</td>
                <td><span class="badge bg-secondary bg-opacity-10 text-secondary">${s.sector || '--'}</span></td>
                <td class="text-center">
                    <button class="btn btn-sm btn-outline-primary view-indicators-btn" data-idx="${idx}" title="View all indicators">
                        <i class="fas fa-chart-bar"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        document.getElementById('resultCount').textContent = `${res.total} results`;

        // Pagination info
        const start = (res.page - 1) * res.per_page + 1;
        const end = Math.min(res.page * res.per_page, res.total);
        document.getElementById('paginationInfo').textContent = `Showing ${start}-${end} of ${res.total}`;

        // Pagination buttons
        renderPagination(res.page, res.pages);

        // Attach indicator button listeners
        document.querySelectorAll('.view-indicators-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const idx = parseInt(btn.dataset.idx);
                showIndicatorModal(allResults[idx]);
            });
        });

    } catch (err) {
        body.innerHTML = '<tr><td colspan="11" class="text-center py-5 text-danger">Error loading data. Please try again.</td></tr>';
        console.error('Screener error:', err);
    }
}

function showIndicatorModal(stock) {
    if (!stock || !stock.indicators) return;

    const modal = new bootstrap.Modal(document.getElementById('indicatorModal'));
    document.getElementById('modalTitle').innerHTML = `<i class="fas fa-chart-bar me-2"></i>${stock.symbol} — Indicator Scorecard`;

    const indicators = stock.indicators;
    const catScores = stock.categoryScores || {};

    // Group indicators by category
    const groups = {};
    for (const [key, ind] of Object.entries(indicators)) {
        const cat = ind.category;
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push({ key, ...ind });
    }

    let html = '';

    // Overall summary card
    html += `
    <div class="indicator-summary mb-4">
        <div class="row align-items-center">
            <div class="col-md-3 text-center">
                <div class="overall-score-circle ${scoreClass(stock.overallScore)}">
                    ${stock.overallScore ? stock.overallScore.toFixed(1) : 'N/A'}
                </div>
                <div class="mt-2 fw-bold">${scoreLabel(stock.overallScore)}</div>
                <div class="text-muted small">${stock.totalIndicators || 0} indicators scored</div>
            </div>
            <div class="col-md-9">
                <h6 class="mb-3">Category Breakdown</h6>
                <div class="row g-2">
                    ${CATEGORY_ORDER.filter(c => catScores[c] !== undefined).map(cat => `
                        <div class="col-md-4 col-6">
                            <div class="cat-score-item">
                                <div class="d-flex justify-content-between align-items-center mb-1">
                                    <span class="small"><i class="${CATEGORY_ICONS[cat] || 'fas fa-circle'} me-1"></i>${cat}</span>
                                    <span class="score-badge-sm ${scoreClass(catScores[cat])}">${catScores[cat] ? catScores[cat].toFixed(1) : '--'}</span>
                                </div>
                                <div class="cat-score-bar">
                                    <div class="cat-score-fill ${scoreClass(catScores[cat])}" style="width:${catScores[cat] ? (catScores[cat] / 4 * 100) : 0}%"></div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    </div>`;

    // Indicator tables by category
    CATEGORY_ORDER.forEach(cat => {
        if (!groups[cat]) return;
        const items = groups[cat];
        html += `
        <div class="indicator-category-section mb-3">
            <h6 class="indicator-cat-title">
                <i class="${CATEGORY_ICONS[cat] || 'fas fa-circle'} me-2"></i>${cat}
                <span class="score-badge-sm ${scoreClass(catScores[cat])} ms-2">${catScores[cat] ? catScores[cat].toFixed(1) : '--'}</span>
            </h6>
            <table class="table table-sm indicator-table mb-0">
                <thead>
                    <tr>
                        <th>Indicator</th>
                        <th class="text-end">Value</th>
                        <th class="text-center">Score</th>
                        <th class="text-center">Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    ${items.map(ind => {
                        const s = ind.score;
                        const sClass = scoreClass(s);
                        const riskLbl = s ? scoreLabel(s) : 'N/A';
                        return `
                        <tr>
                            <td>${ind.name}</td>
                            <td class="text-end fw-semibold">${ind.display || '--'}</td>
                            <td class="text-center">${s !== null && s !== undefined ? `<span class="score-cell ${sClass}">${s}</span>` : '<span class="text-muted">--</span>'}</td>
                            <td class="text-center"><span class="risk-label ${sClass}">${riskLbl}</span></td>
                        </tr>`;
                    }).join('')}
                </tbody>
            </table>
        </div>`;
    });

    document.getElementById('modalBody').innerHTML = html;
    modal.show();
}

function renderPagination(current, total) {
    const ul = document.getElementById('pagination');
    if (total <= 1) { ul.innerHTML = ''; return; }

    let html = '';
    html += `<li class="page-item ${current === 1 ? 'disabled' : ''}"><a class="page-link" href="#" data-page="${current - 1}">&laquo;</a></li>`;

    const start = Math.max(1, current - 2);
    const end = Math.min(total, current + 2);
    for (let i = start; i <= end; i++) {
        html += `<li class="page-item ${i === current ? 'active' : ''}"><a class="page-link" href="#" data-page="${i}">${i}</a></li>`;
    }

    html += `<li class="page-item ${current === total ? 'disabled' : ''}"><a class="page-link" href="#" data-page="${current + 1}">&raquo;</a></li>`;
    ul.innerHTML = html;

    ul.querySelectorAll('.page-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = parseInt(link.dataset.page);
            if (page >= 1 && page <= total) {
                currentPage = page;
                loadScreener();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    });
}

function resetFilters() {
    if (document.getElementById('filterMarket')) document.getElementById('filterMarket').value = 'PSX';
    document.getElementById('filterSector').value = '';
    document.getElementById('filterMarketCap').value = '';
    document.getElementById('filterRisk').value = '';
    ['filterPriceMin', 'filterPriceMax', 'filterPeMin', 'filterPeMax',
     'filterDivMin', 'filterDivMax', 'filterBetaMin', 'filterBetaMax'].forEach(id => {
        document.getElementById(id).value = '';
    });
    currentSort = { by: 'marketCap', dir: 'desc' };
    currentPage = 1;
    loadScreener();
}
