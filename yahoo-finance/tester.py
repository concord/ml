import numpy as np
import matplotlib.pyplot as plt
from yahoo_finance import Share

yahoo = Share('NDAQ')
hist = yahoo.get_historical('2015-10-10', '2015-10-16') 

print [(day['Open'],day['Close']) for day in hist]
# plt.plot([1,2,3])
# plt.xlabel("Time")
# plt.ylabel("Run length")
# plt.show()