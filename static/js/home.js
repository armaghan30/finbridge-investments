/* FINBRIDGE Home page JS - PSX Only */

document.addEventListener('DOMContentLoaded', () => {
    loadMarketOverview();
    loadPopularStocks();
});

async function loadMarketOverview() {
    try {
        const data = await Utils.fetchJSON('/api/market-overview?market=PSX');

        const gBody = document.getElementById('gainersBody');
        if (gBody && data.gainers) {
            gBody.innerHTML = data.gainers.map(s => stockRow(s)).join('');
        }
        const lBody = document.getElementById('losersBody');
        if (lBody && data.losers) {
            lBody.innerHTML = data.losers.map(s => stockRow(s)).join('');
        }
        const aBody = document.getElementById('activeBody');
        if (aBody && data.mostActive) {
            aBody.innerHTML = data.mostActive.map(s => stockRow(s)).join('');
        }
    } catch (err) {
        console.error('Market overview error:', err);
    }
}

async function loadPopularStocks() {
    const container = document.getElementById('popularStocks');
    if (!container) return;

    const popular = ['OGDC', 'PPL', 'MCB', 'HBL', 'LUCK', 'ENGRO', 'FFC', 'NESTLE'];
    const html = [];

    for (const sym of popular) {
        try {
            const q = await Utils.fetchJSON(`/api/stock-quote?ticker=${sym}`);
            const cls = Utils.changeClass(q.changePercent);
            html.push(`
                <div class="col-lg-3 col-md-4 col-sm-6">
                    <a href="/stocks/${q.symbol}" class="text-decoration-none">
                        <div class="popular-stock-card">
                            <div>
                                <div class="symbol">${q.symbol}</div>
                                <div class="name">${q.name}</div>
                            </div>
                            <div class="text-end">
                                <div class="price">${Utils.formatPrice(q.price)}</div>
                                <div class="change ${cls}">${Utils.changeArrow(q.changePercent)}${Utils.formatPercent(q.changePercent)}</div>
                            </div>
                        </div>
                    </a>
                </div>
            `);
        } catch (e) { /* skip */ }
    }
    container.innerHTML = html.join('');
}
