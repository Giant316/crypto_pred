import pandas as pd
import grequests # to make asynchronous HTTP Requests 
from itertools import chain
from datetime import datetime, timedelta
import time 
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

'''
Web Scrapping Script to retrieve intraday hourly trading details
from cryptocompare.com using API:
https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistohour
'''

start_date = datetime(2021, 4, 19, 12, 0)
start_ts = datetime.timestamp(start_date)

# calculate the previous 2000 entries timestamp 
entries = 2000
ts_counter = lambda ts:datetime.timestamp(datetime.fromtimestamp(ts)-timedelta(hours=entries))

# A recursive generator to output previous 2000 entries timestamp recursively
def ts_generator(ts_start):
    yield ts_counter(ts_start)
    yield from ts_generator(ts_counter(ts_start))

# calculate get the timestamp of the earliest n entries by iterating through the infinite generator
# to retrieve data earlier as 2015 (when ETC launched) -> 365*5.5*24/2000 ~24 (set loop to 25)
ts_list = []
gen = ts_generator(start_ts)
for i in range(25):
    ts_list.append(next(gen)) 

base_url = 'https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit={}'.format(entries)
urls = [base_url + "&toTs={}".format(ts) for ts in ts_list]

# induce delay in making request to avoid exceeding quota 
reqs = []
for i in range(0, 25, 5):
    reqs.append((grequests.get(url) for url in urls[i:i+5]))
    time.sleep(5)
 
responses = list(chain(*[grequests.map(req) for req in reqs])) # flatten the nested responses
data = [response.json()['Data']['Data'] for response in responses]
df = pd.DataFrame(list(chain(*data))).drop_duplicates().sort_values(by=['time'])
df['timestamp'] = df.time
df['time'] = [datetime.fromtimestamp(t).strftime("%m/%d/%Y, %H:%M:%S") for t in list(df.time)] # convert timestamp to readable time format
df = df[['time', 'timestamp', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close']]

# drop rows where all columns of interest equal to zero
df = df[df[['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']].eq(0).sum(axis=1) == 0]
df.reset_index(drop=True, inplace=True) # reset index after sorting & dropping rows