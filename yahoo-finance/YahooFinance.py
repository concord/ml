import numpy as np
import matplotlib.pyplot as plt
from yahoo_finance import Share

class Finance:

	def __init__(self, ticker, startDate, endDate):
		"""Creates wrapper instance of yahoo_finance Share class.
		   :param ticker: official ticker for the company
		   :params startDate, endDate: %y-%m-%d, determines which days to get stock price, inclusive
		"""
		self.ticker = ticker
		self._share = Share(self.ticker)
		self._historicalVerbose = self._share.get_historical(startDate,endDate)
		self._historical = self._processData("both")
		self._close = self._processData("close")
		self._open = self._processData("open")
		self._iterator = Stack(self._historical) 

	def _processData(self,includes="both"):
		""" Parse the open and close prices from _historicalVerbose """
		if includes == "both":
			return [(day['Open'],day['Close']) for day in self._historicalVerbose]
		if includes == "close"
			return [day['Close'] for day in self._historicalVerbose]
		if includes == "open":
			return [day['Open'] for day in self._historicalVerbose]

	def microBatch(self):
		""" Return 1 Day of data """
		return self._iterator.pop() if not self._iterator.isEmpty() else 'EMPTY'

	def macroBatch(self, includes="both"):
		""" Returns list of all open/close stock prices from startDate to endDate """
		if includes == "both":
			return self._historical
		if includes == "close":
			return self._close
		if includes == "open": 
			return self._open

	def getRaw(self):
		"""Return raw output from Yahoo Finance API"""
		return self._historicalVerbose

class Stack:
	def __init__(self, data):
		self.__data = data

	def isEmpty(self):
		return len(self.__data) == 0

	def pop(self):
		return self.__data.pop()



