import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
from configs import tickers_list, train_start, train_stop, test_start, test_stop


df = yf.download(tickers_list, train_start, train_stop)['Adj Close']
df = df.dropna()
st = df.index[0]

Y = df.pct_change()
Y = Y.iloc[1:, :]

port = rp.HCPortfolio(returns=Y)
model='HRP' 
codependence = 'pearson' 
rm = 'MV'
rf = 0 
linkage = 'ward' 
max_k = 10 
leaf_order = True 

w = port.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)

portfolio_weights = w

df = yf.download(tickers_list, test_start, test_stop)['Adj Close']

w = round(portfolio_weights['weights'],4)

ret = 0
for i in range(len(portfolio_weights.index)):
    start = df.iloc[0,i]
    end = df.iloc[-1,i]
    gain = end/start
    ret += gain*w[i]
    
    
ret = round(100*(ret-1)/1, 2)
print('')
print(f'Training Period: {st} - {train_stop}')
print (f"Return For Test Period ({test_start}-{test_stop}):   {ret}%")
print('')
