import datetime as dt
from yahoo_finance import Share


def get_stock_prices(ticker, start_date, end_date=None):
    """Gets stock data from Yahoo Finance between dates, inclusive

    Args:
        ticker (str): The company ticker (e.g. GOOG)
        start_date (datetime.date): date to begin collecting share prices
        end_date (datetime.date): date to end collecting share prices. Default
            to today.
    Returns:
        An iterable of (open, close) price tuples.
        Iterable[(float, float)]
    """
    if end_date is None:
        end_date = dt.date.today()

    shares = Share(ticker)
    prices = shares.get_historical(start_date.isoformat(),
                                   end_date.isoformat())

    for day in prices:
        yield float(day["Open"]), float(day["Close"])
