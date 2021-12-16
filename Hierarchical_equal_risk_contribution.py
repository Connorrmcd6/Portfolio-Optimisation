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
model='HERC' 
codependence = 'pearson' 
rm = 'MV'
rf = 0 
linkage = 'ward' 
max_k = 15 
leaf_order = True 

w = port.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)

portfolio_weights = w

for i in range(len(portfolio_weights.index)):
    v = round(portfolio_weights.weights[i],4)
    if v > 0:
        print(f"{portfolio_weights.index[i]}: {100*v}%\n")

weg = round(portfolio_weights['weights'],4)

ret = 0
for i in range(len(portfolio_weights.index)):
    start = df.iloc[0,i]
    end = df.iloc[-1,i]
    gain = end/start
    ret += gain*weg[i]
    
    
ret = round(100*(ret-1)/1, 2)
ret_tr = round((ret/len(df.index))*252,2)


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
print (f"Annual Return For Train Period ({st}-{train_stop}):   {ret_tr}%")
print (f"Return For Test Period ({test_start}-{test_stop}):   {ret}%")
print('')
