from yahoo_finance import Share

def get_data(ticker, start_date, end_date):
	"""Creates wrapper instance of yahoo_finance Share class.
		   :param ticker: official ticker for the company
		   :params startDate, endDate: Datetime.date objects, determines which days to get stock price, inclusive
	"""
	shares = Share(ticker)
	prices = shares.get_historical(start_date.isoformat(),end_date.isoformat())

	for day in prices:
		yield day["Open"], day["Close"]
