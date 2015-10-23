from yahoo_finance import Share
import datetime as dt


def get_stock_prices(ticker, start_date, end_date=dt.date.today()):
    """Gets stock data from Yahoo Finance between dates, inclusive

    Args:
        ticker (str): company ticker, i.e. GOOG
        start_date (datetime.date): date to begin collecting share prices
        end_date (datetime.date):  date to end collecting share prices
    Returns:
        Iterable[(float, float)]: Iterator that returns a tuple
        containing (open,close) share price
    """
    shares = Share(ticker)
    prices = shares.get_historical(start_date.isoformat(),
                                   end_date.isoformat())

    for day in prices:
        yield day["Open"], day["Close"]
