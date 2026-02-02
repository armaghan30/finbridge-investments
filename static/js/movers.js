/* Movers page JS */

document.addEventListener('DOMContentLoaded', () => {
    loadMovers();
});

async function loadMovers() {
    const body = document.getElementById('moversBody');
    if (!body) return;

    try {
        const data = await Utils.fetchJSON('/api/market-overview?market=PSX');

        let items;
        if (MOVER_TYPE === 'gainers') items = data.gainers;
        else if (MOVER_TYPE === 'losers') items = data.losers;
        else items = data.mostActive;

        if (!items || !items.length) {
            body.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-muted">No data available</td></tr>';
            return;
        }

        body.innerHTML = items.map((s, i) => {
            const cls = Utils.changeClass(s.changePercent);
            return `<tr>
                <td>${i + 1}</td>
                <td><a href="/stocks/${s.symbol}" class="stock-link">${s.symbol}</a></td>
                <td>${s.name}</td>
                <td class="text-end">${Utils.formatPrice(s.price)}</td>
                <td class="text-end ${cls}">${Utils.changeArrow(s.change)}${s.change >= 0 ? '+' : ''}${Number(s.change).toFixed(2)}</td>
                <td class="text-end ${cls}">${Utils.formatPercent(s.changePercent)}</td>
                <td class="text-end">${Utils.formatNumber(s.volume)}</td>
                <td class="text-end">${Utils.formatNumber(s.marketCap)}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        body.innerHTML = '<tr><td colspan="8" class="text-center py-4 text-danger">Error loading data</td></tr>';
    }
}
