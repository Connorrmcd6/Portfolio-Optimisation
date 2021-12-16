import pandas as pd
import numpy as np
import yfinance as yf
import math
from scipy.optimize import minimize
from configs import tickers_list, train_start, train_stop, test_start, test_stop


df = yf.download(tickers_list, train_start, train_stop)['Adj Close']
df = df.dropna()

returns = df/df.shift(1)
returns = returns.iloc[1:,:]
log_returns = np.log(returns)

mean_log_returns = log_returns.mean()
sigma = log_returns.cov()

def negative_sharpe(w):
    w = np.array(w)
    R = sum(mean_log_returns*w)
    V = np.sqrt(np.dot(w.T,np.dot(sigma,w)))
    s = R/V
    return -s

def sum_to_one(w):
    return sum(w)-1

w0 = []
bounds = []
for i in range(len(df.columns)):
    if i != len(df.columns):
        w0.append(1/len(df.columns))
        bounds.append((0,1))
    
    else:
        w0.append(math.floor(1/len(df.columns)))
        bounds.append((0,1))

constraints = ({'type':'eq', 'fun':sum_to_one})

w_opt = minimize(negative_sharpe, w0, method = 'SLSQP', bounds = bounds, constraints = constraints)

#final portfolio allocation
portfolio_weights = pd.DataFrame(w_opt.x)
portfolio_weights.index = returns.columns
portfolio_weights.columns = ['optimise_weights']

for i in range(len(portfolio_weights['optimise_weights'])):
    x = round(100*portfolio_weights['optimise_weights'][i],2)
    if x > 0:
        print(portfolio_weights.index[i])
        print(f'{x}%')
        print('')


df = yf.download(tickers_list, test_start, test_stop)['Adj Close']



w = round(portfolio_weights['optimise_weights'],4)

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
