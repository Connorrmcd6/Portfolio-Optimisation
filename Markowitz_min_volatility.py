import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from configs import tickers_list, train_start, train_stop, test_start, test_stop


df = yf.download(tickers_list, train_start, train_stop)['Adj Close']
df = df.dropna()

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

ef = EfficientFrontier(mu, S)
weights = ef.min_volatility()

p_weights = []
for i in weights:
    p_weights.append(round(weights[i], 4))
    v = round(100*weights[i],2)
    if  v > 0:
        print(f"{i}: {v}%\n")

print('Training Period Perfomance:')
ef.portfolio_performance(verbose=True)


portfolio_weights = pd.DataFrame(p_weights)
portfolio_weights.index = df.columns
portfolio_weights.columns = ['weights']


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
print (f"Return For Test Period: {ret}%")
print('')


