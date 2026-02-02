from flask import Blueprint, render_template, request

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    return render_template('pages/home.html')

@main_bp.route('/screener')
def screener():
    return render_template('pages/screener.html')

@main_bp.route('/stocks/<ticker>')
def stock_detail(ticker):
    tab = request.args.get('tab', 'overview')
    return render_template('pages/stock_detail.html', ticker=ticker.upper(), active_tab=tab)

@main_bp.route('/smif')
def smif():
    return render_template('pages/smif.html')

@main_bp.route('/learn')
def learn():
    return render_template('pages/learn.html')

@main_bp.route('/scholars-program')
def scholars_program():
    return render_template('pages/scholars_program.html')

@main_bp.route('/about')
def about():
    return render_template('pages/about.html')

@main_bp.route('/markets')
def markets():
    return render_template('pages/markets.html')

@main_bp.route('/gainers')
def gainers():
    return render_template('pages/movers.html', mover_type='gainers')

@main_bp.route('/losers')
def losers():
    return render_template('pages/movers.html', mover_type='losers')

@main_bp.route('/most-active')
def most_active():
    return render_template('pages/movers.html', mover_type='active')

@main_bp.route('/ipo-calendar')
def ipo_calendar():
    return render_template('pages/ipo.html')

@main_bp.route('/news')
def news():
    return render_template('pages/news.html')
